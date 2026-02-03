from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.types import Command

from agent.tools import retrieve_tables, retrieve_wiki_passages
from agent.utils import State, llm


# Analysis-Agent system prompt
ANALYSIS_AGENT_SYSTEM_PROMPT = (
    "Analysis agent: analyze table_agent data and produce the final answer. Tables are HybridQA (table_uid). "
    "Only you can find tables (retrieve_tables) or get Wikipedia context (retrieve_wiki_passages).\n\n"
    "Responsibilities: (1) Decide if current data (tables + wiki passages) is sufficient to answer the question. "
    "(2) If data is INSUFFICIENT you MUST retrieve more: call retrieve_tables(query) when more or different tables might "
    "contain the answer—then in your response tell the planner which table_uid(s) the table_agent should use and what to "
    "extract; call retrieve_wiki_passages(query) when you need definitions, background, or general knowledge from Wikipedia. "
    "Use one or both depending on the gap (more table content vs external knowledge). (3) When data is sufficient, compute "
    "metrics and produce JSON (reasoning, functions_called, metrics, answer, sources: table_uid, cell/row refs, wiki passages).\n\n"
    "Tools: retrieve_tables(query)—returns table UIDs by relevance; call only after you have evaluated current data. "
    "retrieve_wiki_passages(query)—returns Wikipedia passage texts; call when external knowledge would help answer the question.\n\n"
    "Workflow: (1) Review the question and all prior data (table_agent extractions, any previous tool results). (2) Sufficient? "
    "→ analyze and output final JSON. (3) Not sufficient? → you MUST call retrieve_tables and/or retrieve_wiki_passages; then "
    "if you requested table extractions, specify table_uid(s) and what to extract and stop (no final answer yet)—planner will "
    "route to table_agent and you will run again with new data. (4) When sufficient, analyze all data and output explanation + JSON.\n\n"
    "Final JSON: {\"reasoning\": \"...\", \"functions_called\": [...], \"metrics\": {...}, \"answer\": \"...\", \"sources\": [...]}. "
    "ReAct: Think (sufficient?) → Act (if insufficient, must call retrieve_tables and/or retrieve_wiki_passages; request extractions if needed) → Observe and re-evaluate."
)

analysis_agent = create_agent(llm, tools=[retrieve_tables, retrieve_wiki_passages])

def analysis_node(state: State) -> Command[Literal["router"]]:
    """Analysis agent node that performs calculations and generates JSON-formatted responses."""
    # Add system prompt to the state
    messages_with_system = [SystemMessage(content=ANALYSIS_AGENT_SYSTEM_PROMPT)] + state["messages"]
    state_with_system = {**state, "messages": messages_with_system}
    
    result = analysis_agent.invoke(state_with_system)
    
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="analysis_agent")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="router",
    )
