from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from agent.utils import State

from agent.planner import answer_supervisor_node
from agent.analysis_agent import analysis_node
from agent.table_agent import table_agent_node
from agent.utils import validate_input, route

# Define the graph
graph = (
    StateGraph(state_schema=State, input_schema=MessagesState)
    .add_node("validator", validate_input)
    .add_node("router", answer_supervisor_node)
    .add_node("table_agent", table_agent_node)
    .add_node("analysis_agent", analysis_node)
    .add_edge(START, "validator")
    .add_conditional_edges(
        "validator",
        route,
        {
            "router": "router",
            "end": END,
        },
    )
    .compile(name="New Graph")
)
