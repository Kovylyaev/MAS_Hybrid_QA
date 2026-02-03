from __future__ import annotations

from typing import Literal

from langgraph.graph import END
from langchain_core.messages import SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from agent.utils import State, llm


def make_planner_node(llm: BaseChatModel, members: list[str]):
    options = ["FINISH"] + members
    system_prompt = (
        "Planner: break the user request into steps and route between workers. Tables are from HybridQA (table_uid). "
        "Only analysis_agent can find tables (retrieve_tables) or get wikipedia passages (retrieve_wiki_passages); table_agent only extracts when given a table_uid.\n\n"
        f"Workers:\n"
        f"- {members[0]}: Extract table data only (get_table_metadata, find_rows_by_value, get_cell, get_column, get_row_by_index). "
        f"Needs a table_uid from you or analysis_agent. Returns JSON with extracted data.\n"
        f"- {members[1]}: Think, analyze, evaluate. Has retrieve_tables(query) and retrieve_wiki_passages(query). "
        f"When data is insufficient it MUST call retrieve_tables and/or retrieve_wiki_passages; then produces final JSON "
        f"(reasoning, functions_called, metrics, answer, sources).\n\n"
        "Workflow: (1) Route to analysis_agent first. (2) If it requests extractions, route to table_agent with table_uid and task. "
        "(3) When table_agent returns, route back to analysis_agent. (4) When analysis_agent gives the final answer, respond FINISH.\n\n"
        "ReAct: Reason → choose table_agent (extraction) or analysis_agent (evaluation/retrieval/analysis) → observe results. "
        "If analysis_agent says data is insufficient and names table_uid(s), route to table_agent next."
    )

    class Router(BaseModel):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: str = Field(description=f"Must be one of: {', '.join(options)}")
        
        @field_validator('next')
        @classmethod
        def validate_next(cls, v):
            if v not in options:
                raise ValueError(f"next must be one of {options}, got {v}")
            return v

    def planner_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router that uses ReAct-style reasoning to break down tasks."""
        # Add system prompt as SystemMessage
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response.next
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})


    return planner_node

answer_supervisor_node = make_planner_node(llm, ["table_agent", "analysis_agent"])
