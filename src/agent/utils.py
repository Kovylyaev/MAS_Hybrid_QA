from __future__ import annotations

from typing import Any, Dict, List
import os
from datasets import load_dataset
from pathlib import Path
import pickle
from git import Repo, RemoteProgress
import json

from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import WikipediaRetriever

class State(MessagesState):
    next: str


llm = ChatOpenAI(model="gpt-4o")

class Storage:
    """In-memory storage and retriever for HybridQA tables.

    This class loads table data from the ``wenhu/hybrid_qa`` dataset into memory
    and builds a vector-based retriever over table identifiers. It exposes
    convenience methods for:

    - Accessing a table by its UID
    - Retrieving relevant table UIDs for a natural-language query
    """
    def __init__(self):
        print("\n\n\n Initializing Storage \n\n\n")

        self.data_dir = Path("./data")
        self.table_vectorstore_dir = self.data_dir / "table_vector_store"
        self.wiki_vectorstore_dir = self.data_dir / "wiki_vector_store"
        self.tables_dir = self.data_dir / "tables.pkl"
        self.passages_dir = self.data_dir / "passages.pkl"

        self.tables = {}

        # Загружаем или создаём и сохраняем векторное хранилище с эмбедингами
        if (
            self.data_dir.exists()
            and self.table_vectorstore_dir.exists() 
            and self.wiki_vectorstore_dir.exists() 
            and self.tables_dir.exists() 
            and self.passages_dir.exists()
        ):
            table_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)
            table_vectorstore = InMemoryVectorStore.load(str(self.table_vectorstore_dir), table_embeddings)

            wiki_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)
            wiki_vectorstore = InMemoryVectorStore.load(str(self.wiki_vectorstore_dir), wiki_embeddings)

            with open(str(self.tables_dir), "rb") as f:
                self.tables = pickle.load(f)

            with open(str(self.passages_dir), "rb") as f:
                self.passages = pickle.load(f)
        else:
            if not Path("./data/hybrid_qa").exists():
                class CloneProgress(RemoteProgress):
                    def update(self, op_code, cur_count, max_count=None, message=''):
                        if cur_count and max_count:
                            print(cur_count, " / ", max_count)

                Repo.clone_from(
                    "https://github.com/wenhuchen/WikiTables-WithLinks",
                    "./data/hybrid_qa",
                    progress=CloneProgress()
                )

            table_vectorstore = self._create_table_vectorstore()
            wiki_vectorstore = self._create_wiki_vectorstore()

        self.table_retriever = table_vectorstore.as_retriever()
        self.wiki_retriever = wiki_vectorstore.as_retriever()

        print("\n\n\n Storage is ready \n\n\n")

    def _create_table_vectorstore(self):
        table_uids = set()
        for table_path in Path("./data/hybrid_qa/tables_tok").iterdir():
            self.tables[table_path.stem] = table_path
            table_uids.add(table_path.stem)
        table_uids = list(table_uids)
            
        print("\n\n\n Creating Table VectorStore \n\n\n")
        embeddings = OpenAIEmbeddings(chunk_size=1000, model="text-embedding-3-small", dimensions=256)
        vectorstore = InMemoryVectorStore.from_texts(table_uids, embedding=embeddings)
        print("\n\n\n Table VectorStore is created \n\n\n")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(str(self.tables_dir), "wb") as f:
            pickle.dump(self.tables, f)
        vectorstore.dump(str(self.table_vectorstore_dir))

        return vectorstore

    def _create_wiki_vectorstore(self):
        self.passages = {}
        for passages_path in Path("./data/hybrid_qa/request_tok").iterdir():
            with open(passages_path, "r") as f:
                self.passages.update(json.load(f))
            
        print("\n\n\n Creating Wiki-Passage VectorStore \n\n\n")
        embeddings = OpenAIEmbeddings(chunk_size=1000, model="text-embedding-3-small", dimensions=256)
        vectorstore = InMemoryVectorStore.from_texts(self.passages.keys(), embedding=embeddings)
        print("\n\n\n Wiki-Passage VectorStore is created \n\n\n")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(str(self.passages_dir), "wb") as f:
            pickle.dump(self.passages, f)
        vectorstore.dump(str(self.wiki_vectorstore_dir))

        return vectorstore


    def get_table(self, table_uid: str) -> Dict[str, Any]:
        """Retrieve a table by its ID.
        
        Args:
            table_uid: The identifier of the table to retrieve.
        
        Returns:
            A dictionary containing the table data (title, header, data, etc.).
        
        Raises:
            ValueError: If the table_uid is not found.
        """
        if table_uid not in self.tables:
            raise ValueError(f"Table with uid: '{table_uid}' not found")

        with open(self.tables[table_uid], "r") as f:
            return json.load(f)

    def retrieve_tables(self, query: str) -> list[str]:
        """Retrieve candidate table UIDs relevant to a natural-language query.

        This method uses the underlying vector retriever to find the most
        similar table identifiers to the provided query. It is intended to
        narrow down which tables should be inspected by the table agent.

        Args:
            query: Natural-language question or description of the information
                the user is looking for.

        Returns:
            A list of table UIDs ordered by relevance to the query.
        """
        retrieved_documents = self.table_retriever.invoke(query)

        table_uids = []
        for doc in retrieved_documents:
            table_uids.append(doc.page_content)

        return table_uids

    def retrieve_wiki_passages(self, query: str) -> list[str]:
        """Retrieve Wikipedia passage texts relevant to a natural-language query.

        This method uses the Wikipedia retriever to fetch up to a fixed number of
        Wikipedia document excerpts whose content is most relevant to the query.
        Useful for adding external context when table data alone is insufficient
        to answer a question.

        Args:
            query: Natural-language question or description of the information
                the user is looking for.

        Returns:
            A list of passage text strings, ordered by relevance to the query.
            The number of passages is limited by the retriever configuration
            (e.g. load_max_docs).
        """
        retrieved_documents = self.wiki_retriever.invoke(query)

        passages_texts = []
        for doc in retrieved_documents:
            passages_texts.append(self.passages[doc.page_content])

        return passages_texts

STORAGE = None

def get_storage():
    global STORAGE
    if STORAGE is None:
        STORAGE = Storage()
    return STORAGE


def validate_input(input_state: MessagesState) -> State:
    if not input_state["messages"]:
        return {
            "messages": [
                AIMessage(content="Please enter a question")
            ],
            "next": "end",
        }
    return {"messages": input_state["messages"], "next": "router",}

def route(state):
    return state["next"]