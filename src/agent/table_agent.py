from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.types import Command

from agent.tools import (
    get_table_metadata,
    find_rows_by_value,
    get_cell,
    get_column,
    get_row_by_index,
)
from agent.utils import State, llm


# Table-Tool Agent system prompt
TABLE_AGENT_SYSTEM_PROMPT = (
    "Table agent: extract data from tables only. No analysis or evaluation—analysis_agent does that. "
    "Tables are HybridQA, keyed by table_uid. You have no search/list tool; planner or analysis_agent will give you table_uid(s).\n\n"
    "Tools: get_table_metadata(table_uid), find_rows_by_value(table_uid, conditions), get_cell(...), get_column(...), get_row_by_index(...).\n\n"
    "Workflow: (1) Require a table_uid; if missing, ask for it. (2) Use get_table_metadata to see structure. "
    "(3) Extract what was requested (cells, rows, columns, row indices). (4) Report only what you extracted.\n\n"
    "Respond in JSON only: {\"functions_called\": [...], \"table_uid\": \"...\", \"extracted_data\": {cells/rows/columns/row_indices/metadata}, \"summary\": \"...\"}. "
    "Do not evaluate answers or reason about the question; if no table_uid, ask for it."
)

table_agent = create_agent(
    llm,
    tools=[
        get_table_metadata,
        find_rows_by_value,
        get_cell,
        get_column,
        get_row_by_index,
    ],
)

def table_agent_node(state: State) -> Command[Literal["router"]]:
    """Table agent node that adds system prompt and processes table-related queries."""
    # Add system prompt to the state
    messages_with_system = [SystemMessage(content=TABLE_AGENT_SYSTEM_PROMPT)] + state["messages"]
    state_with_system = {**state, "messages": messages_with_system}
    
    result = table_agent.invoke(state_with_system)
    
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="table_agent")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="router",
    )
