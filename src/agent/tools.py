from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.tools import tool

from agent.utils import get_storage

@tool
def get_table_metadata(table_uid: str) -> {
    "ok": bool,
    "table title": str,
    "columns": list[str],
    "num rows": int,
}:
    """Get metadata about a table including its title, column names, and row count.
    
    This tool retrieves high-level information about a table without loading
    the full table data. Useful for understanding the structure of a table
    before querying specific data.
    
    Args:
        table_uid: The UID of the table to get metadata for.
    
    Returns:
        On success (`ok == True`):
            - "ok": True
            - "table title": The title/name of the table
            - "columns": List of column names in the table
            - "num rows": Total number of data rows in the table
        On failure (`ok == False`):
            - "ok": False
            - "error": Error message describing what went wrong
            - "table_uid": The UID that caused the error
    
    Example:
        >>> result = get_table_metadata("table_123")
        >>> if result["ok"]:
        ...     print(result["columns"])
        ... else:
        ...     print("Error:", result["error"])
    """
    try:
        table = get_storage().get_table(table_uid)
        return {
            "ok": True,
            "table title": table["title"],
            "columns": [header[0] for header in table["header"]],
            "num rows": len(table["data"]),
        }
    except ValueError as e:
        return {
            "ok": False,
            "error": str(e),
            "table_uid": table_uid,
        }

def _get_column_index(table: Dict[str, Any], column_name: str) -> int:
    """Helper function to get the index of a column by name.
    
    Args:
        table: The table dictionary containing header information.
        column_name: The name of the column to find.
    
    Returns:
        The zero-based index of the column.
    
    Raises:
        ValueError: If the column name is not found in the table header.
    """
    for i, name in enumerate([header[0] for header in table["header"]]):
        if name == column_name:
            return i
    raise ValueError(f"Column '{column_name}' not found")

@tool
def get_column(table_uid: str, column_name: str) -> {
    "ok": bool,
    "cells": list[str],
}:
    """Retrieve all values from a specific column in a table.
    
    This tool extracts all cell values from a single column across all rows.
    Useful for analyzing a specific attribute or field across the entire dataset.
    
    Args:
        table_uid: The UID of the table to query.
        column_name: The name of the column to retrieve.
    
    Returns:
        On success (`ok == True`):
            - "ok": True
            - "cells": List of all cell values in the specified column
        On failure (`ok == False`), e.g. table or column not found:
            - "ok": False
            - "error": Error message describing what went wrong
            - "table_uid": The UID that caused the error
            - "column_name": The column name that caused the error
    
    Example:
        >>> result = get_column("table_123", "Name")
        >>> if result["ok"]:
        ...     print(result["cells"])
        ... else:
        ...     print("Error:", result["error"])
    """
    try:
        table = get_storage().get_table(table_uid)

        column_index = _get_column_index(table, column_name)

        values = [row[column_index][0] for row in table["data"]]

        return {
            "ok": True,
            "cells": values
        }
    except ValueError as e:
        return {
            "ok": False,
            "error": str(e),
            "table_uid": table_uid,
            "column_name": column_name,
        }

@tool
def get_row_by_index(table_uid: str, row_index: int) -> {
    "ok": bool,
    "row": list[str],
}:
    """Retrieve a complete row from a table by its index.
    
    This tool returns all cell values from a specific row. Useful for getting
    all information about a single record or entity in the table.
    
    Args:
        table_uid: The UID of the table to query.
        row_index: The zero-based index of the row to retrieve.
    
    Returns:
        On success (`ok == True`):
            - "ok": True
            - "row": List of all cell values in the specified row
        On failure (`ok == False`), e.g. table not found:
            - "ok": False
            - "error": Error message describing what went wrong
            - "table_uid": The UID that caused the error
            - "row_index": The requested row index
    
    Example:
        >>> result = get_row_by_index("table_123", 0)
        >>> if result["ok"]:
        ...     print(result["row"])
        ... else:
        ...     print("Error:", result["error"])
    """
    try:
        table = get_storage().get_table(table_uid)

        values = [cell[0] for cell in table["data"][row_index]]

        return {
            "ok": True,
            "row": values
        }
    except ValueError as e:
        return {
            "ok": False,
            "error": str(e),
            "table_uid": table_uid,
            "row_index": row_index,
        }


@tool
def get_cell(table_uid: str, row_index: int, column_name: str) -> {
    "ok": bool,
    "cell": str,
}:
    """Retrieve a single cell value from a table.
    
    This tool gets the value of a specific cell at the intersection of a row
    and column. Useful for precise data extraction when you know both the
    row position and column name.
    
    Args:
        table_uid: The UID of the table to query.
        row_index: The zero-based index of the row containing the cell.
        column_name: The name of the column containing the cell.
    
    Returns:
        On success (`ok == True`):
            - "ok": True
            - "cell": The value of the specified cell as a string
        On failure (`ok == False`), e.g. table or column not found:
            - "ok": False
            - "error": Error message describing what went wrong
            - "table_uid": The UID that caused the error
            - "column_name": The column name that caused the error
    
    Example:
        >>> result = get_cell("table_123", 0, "Name")
        >>> if result["ok"]:
        ...     print(result["cell"])
        ... else:
        ...     print("Error:", result["error"])
    """
    try:
        table = get_storage().get_table(table_uid)

        column_index = _get_column_index(table, column_name)

        value = table["data"][row_index][column_index][0]

        return {
            "ok": True,
            "cell": value
        }
    except ValueError as e:
        return {
            "ok": False,
            "error": str(e),
            "table_uid": table_uid,
            "column_name": column_name,
        }


@tool
def find_rows_by_value(table_uid: str, conditions: dict[str, str]) -> {
    "ok": bool,
    "row indices": list[int],
}:
    """Find rows in a table that match specific column-value conditions.
    
    This tool searches through a table and returns the indices of rows where
    all specified conditions are met. Conditions are specified as a dictionary
    mapping column names to their expected values. A row matches only if ALL
    conditions are satisfied.
    
    Args:
        table_uid: The UID of the table to search.
        conditions: A dictionary mapping column names to their expected values.
                    All conditions must be met for a row to be included.
    
    Returns:
        On success (`ok == True`):
            - "ok": True
            - "row indices": List of zero-based row indices that match all conditions
        On failure (`ok == False`), e.g. table or column not found:
            - "ok": False
            - "error": Error message describing what went wrong
            - "table_uid": The UID that caused the error
            - "conditions": The conditions that were used in the query
    
    Example:
        >>> result = find_rows_by_value("table_123", {"City": "New York", "Age": "25"})
        >>> if result["ok"]:
        ...     print(result["row indices"])
        ... else:
        ...     print("Error:", result["error"])
    """
    try:
        table = get_storage().get_table(table_uid)

        row_indices = []
        for row_index, row in enumerate(table["data"]):
            for column_name, value in conditions.items():
                column_index = _get_column_index(table, column_name)
                if row[column_index][0] != value:
                    break
                else:
                    row_indices.append(row_index)

        return {
            "ok": True,
            "row indices": row_indices
        }
    except ValueError as e:
        return {
            "ok": False,
            "error": str(e),
            "table_uid": table_uid,
            "conditions": conditions,
        }

@tool
def retrieve_tables(query: str) -> {
    "ok": bool,
    "table uids": list[str],
}:
    """Retrieve candidate tables relevant to a natural-language query.

    This tool uses the underlying vector retriever to find table UIDs whose
    semantics are most similar to the provided query. It is intended to narrow
    down the set of tables that the table_agent should inspect for a given
    user question.

    Args:
        query: Natural-language user question or description of the information
            being sought.

    Returns:
        A dictionary containing:

        - ``ok``: Always ``True`` if the tool returns successfully.
        - ``table uids``: List of table UIDs ordered by relevance to the query.

    Example:
        >>> result = retrieve_tables(\"Which tables talk about World War II?\")
        >>> if result[\"ok\"]:
        ...     print(result[\"table uids\"][:5])
    """
    table_uids = get_storage().retrieve_tables(query)

    return {
        "ok": True,
        "table uids": table_uids,
    }

@tool
def retrieve_wiki_passages(query: str) -> {
    "ok": bool,
    "passages texts": list[str],
}:
    """Retrieve Wikipedia passage texts relevant to a natural-language query.

    This tool uses the storage's Wikipedia retriever to fetch passage excerpts
    whose content is most relevant to the query. Use it when you need additional
    context from Wikipedia to answer a question (e.g. when table data is
    insufficient or you need definitions, background, or related facts).

    Args:
        query: Natural-language question or description of the information
            being sought.

    Returns:
        A dictionary containing:

        - ``ok``: Always ``True`` if the tool returns successfully.
        - ``passages texts``: List of passage text strings, ordered by
          relevance to the query.

    Example:
        >>> result = retrieve_wiki_passages("Who won the 2020 Olympics marathon?")
        >>> if result["ok"]:
        ...     for text in result["passages texts"][:9]:
        ...         print(text[:100])
    """
    passages = get_storage().retrieve_wiki_passages(query)

    return {
        "ok": True,
        "passages texts": passages,
    }