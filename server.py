#!/usr/bin/env python3
"""
Mosaic - Modular Multi-Agent Tools for Python (MCP Server)

A modern toolkit for building, combining, and experimenting with modular multi-agent tools.

Author: Garvit Mehra
Version: 1.0.0
License: MIT
"""

import os
import sqlite3
import logging
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Union

# Third-party imports
from mcp.server.fastmcp import FastMCP
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name="Mosaic Database Server",
    host="127.0.0.1",
    port=8000
)

# Global state
selected_table = None


# =============================================================================
# Data Models
# =============================================================================

class WhereCondition(BaseModel):
    """Model for WHERE clause conditions."""
    column: str
    operator: str
    value: Union[str, int, float, bool, None]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ColumnDefinition(BaseModel):
    """Model for table column definitions."""
    name: str
    type: str
    constraints: Optional[str] = ""


class CreateTableRequest(BaseModel):
    """Model for table creation requests."""
    table: str
    columns: List[ColumnDefinition]


class ViewTableRequest(BaseModel):
    """Model for table viewing requests."""
    select: Optional[str] = "*"
    where: Optional[List[WhereCondition]] = []


class EditTableRequest(BaseModel):
    """Model for table editing requests."""
    table: str
    updates: Dict[str, Union[str, int, float, bool, None]]
    where: List[WhereCondition]


class InsertDataRequest(BaseModel):
    """Model for data insertion requests."""
    data: Dict[str, Union[str, int, float, bool, None]]


class DeleteDataRequest(BaseModel):
    """Model for data deletion requests."""
    where: List[WhereCondition]


class DiscoverColumnsRequest(BaseModel):
    """Model for column discovery requests."""
    table: Optional[str] = None


class FilterCondition(BaseModel):
    """Model for filter conditions."""
    column: str
    value: Union[str, int, float, bool, None]


class FilteredDataRequest(BaseModel):
    """Model for filtered data requests."""
    table: Optional[str] = None
    filters: Optional[List[FilterCondition]] = None


class DropTableRequest(BaseModel):
    """Model for table deletion requests."""
    table: str


# =============================================================================
# Database Utilities
# =============================================================================

def get_database_path() -> str:
    """Get the database file path."""
    return os.path.join(os.path.dirname(__file__), "db.sqlite")


@contextmanager
def safe_db_connection():
    """Context manager for safe database connections."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


def get_db_connection():
    """Create and return a database connection."""
    db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the SQLite database and ensure it exists."""
    try:
        db_path = get_database_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.close()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize database: {str(e)}")


def validate_table_exists(table_name: str) -> bool:
    """Validate that a table exists in the database."""
    with safe_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        exists = cursor.fetchone() is not None
        
    if not exists:
        raise HTTPException(status_code=400, detail=f"Table '{table_name}' does not exist.")
    return True


def validate_columns_exist(table_name: str, columns: List[str]) -> None:
    """Validate that specified columns exist in a table."""
    with safe_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        valid_columns = {col[1] for col in cursor.fetchall()}
        
        for col in columns:
            if col not in valid_columns:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Column '{col}' does not exist in table '{table_name}'."
                )


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool(
    name="create_table",
    description="Create a new table in the SQLite database with specified columns and constraints."
)
async def create_table(*, request: CreateTableRequest) -> Dict[str, str]:
    """Create a new table in the database."""
    try:
        if not request.table or not request.columns:
            raise HTTPException(status_code=400, detail="Table name and columns are required.")
        
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if table already exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (request.table,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail=f"Table '{request.table}' already exists.")
            
            # Build column definitions
            columns = [
                f"{col.name} {col.type} {col.constraints or ''}".strip() 
                for col in request.columns
            ]
            
            # Create table
            query = f"CREATE TABLE {request.table} ({', '.join(columns)})"
            cursor.execute(query)
            conn.commit()
            
            logger.info(f"Created table '{request.table}' with columns: {[col.name for col in request.columns]}")
            return {"message": f"Table '{request.table}' created successfully"}
            
    except Exception as e:
        logger.error(f"Error in create_table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="view_table",
    description="View data from the currently selected table with optional filters and column selection."
)
async def view_table(*, request: ViewTableRequest) -> Dict[str, Any]:
    """View data from the selected table."""
    global selected_table
    
    if not selected_table:
        raise HTTPException(
            status_code=400, 
            detail="No table is currently selected. Use edit_table to select a table first."
        )
    
    validate_table_exists(selected_table)
    
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT {request.select} FROM {selected_table}"
            params = []
            
            # Add WHERE conditions if specified
            if request.where:
                conditions = []
                for cond in request.where:
                    if not hasattr(cond, "column") or not hasattr(cond, "operator") or not hasattr(cond, "value"):
                        raise HTTPException(status_code=400, detail="Invalid where condition format.")
                    conditions.append(f"{cond.column} {cond.operator} ?")
                    params.append(cond.value)
                query += f" WHERE {' AND '.join(conditions)}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            
            logger.info(f"Viewed table '{selected_table}' with {len(result)} rows")
            return {"data": result}
            
    except Exception as e:
        logger.error(f"Error in view_table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="edit_table",
    description="Select a table for editing and optionally update rows based on conditions."
)
async def edit_table(*, request: EditTableRequest) -> Dict[str, str]:
    """Select a table for editing and optionally update data."""
    global selected_table
    
    validate_table_exists(request.table)
    
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            selected_table = request.table
            
            # Update data if specified
            if request.updates:
                validate_columns_exist(request.table, list(request.updates.keys()))
                
                set_clause = ", ".join([f"{key} = ?" for key in request.updates.keys()])
                params = list(request.updates.values())
                query = f"UPDATE {request.table} SET {set_clause}"
                
                # Add WHERE conditions
                if request.where:
                    conditions = [f"{cond.column} {cond.operator} ?" for cond in request.where]
                    query += f" WHERE {' AND '.join(conditions)}"
                    params.extend([cond.value for cond in request.where])
                
                cursor.execute(query, params)
                conn.commit()
                
                logger.info(f"Updated {cursor.rowcount} rows in table '{request.table}'")
                return {
                    "message": f"Updated {cursor.rowcount} rows in '{request.table}'. "
                              f"Table is now selected for further operations."
                }
            else:
                logger.info(f"Selected table '{request.table}' for editing")
                return {"message": f"Table '{request.table}' is now selected for editing."}
                
    except Exception as e:
        logger.error(f"Error in edit_table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="insert_data",
    description="Insert a new row into the currently selected table."
)
async def insert_data(*, request: InsertDataRequest) -> Dict[str, str]:
    """Insert data into the selected table."""
    global selected_table
    
    if not selected_table:
        raise HTTPException(
            status_code=400, 
            detail="No table is currently selected. Use edit_table to select a table first."
        )
    
    if not request.data or not isinstance(request.data, dict):
        raise HTTPException(status_code=400, detail="Data for insertion must be a non-empty dictionary.")
    
    validate_table_exists(selected_table)
    
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            validate_columns_exist(selected_table, list(request.data.keys()))
            
            columns = ", ".join(request.data.keys())
            placeholders = ", ".join(["?" for _ in request.data])
            query = f"INSERT INTO {selected_table} ({columns}) VALUES ({placeholders})"
            
            cursor.execute(query, list(request.data.values()))
            conn.commit()
            
            logger.info(f"Inserted data into '{selected_table}': {request.data}")
            return {"message": f"Inserted row into '{selected_table}'"}
            
    except Exception as e:
        logger.error(f"Error in insert_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="delete_data",
    description="Delete rows from the currently selected table based on conditions."
)
async def delete_data(*, request: DeleteDataRequest) -> Dict[str, str]:
    """Delete data from the selected table."""
    global selected_table
    
    if not selected_table:
        raise HTTPException(
            status_code=400, 
            detail="No table is currently selected. Use edit_table to select a table first."
        )
    
    validate_table_exists(selected_table)
    
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            query = f"DELETE FROM {selected_table}"
            params = []
            
            # Add WHERE conditions if specified
            if request.where:
                conditions = [f"{cond.column} {cond.operator} ?" for cond in request.where]
                query += f" WHERE {' AND '.join(conditions)}"
                params = [cond.value for cond in request.where]
            
            cursor.execute(query, params)
            conn.commit()
            
            logger.info(f"Deleted {cursor.rowcount} rows from '{selected_table}'")
            return {"message": f"Deleted {cursor.rowcount} rows from '{selected_table}'"}
            
    except Exception as e:
        logger.error(f"Error in delete_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="view_database",
    description="Retrieve schema and metadata of all tables in the database."
)
async def view_database() -> Dict[str, Any]:
    """View database schema and metadata."""
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            result = []
            for table in tables:
                table_name = table[0]
                table_sql = table[1]
                
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_details = [
                    {
                        "name": col[1], 
                        "type": col[2], 
                        "notnull": bool(col[3]), 
                        "pk": bool(col[5])
                    }
                    for col in columns
                ]
                
                result.append({
                    "name": table_name, 
                    "sql": table_sql, 
                    "columns": column_details
                })
            
            logger.info("Viewed database schema and metadata")
            return {"tables": result}
            
    except Exception as e:
        logger.error(f"Error in view_database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="traverse_database",
    description="List all tables in the database with their structure and row counts as a formatted string."
)
async def traverse_database() -> str:
    """List all tables with structure and row counts."""
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            output = ["Database Overview:"]
            
            if not tables:
                output.append("No tables found in the database.")
                return "\n".join(output)
            
            for table in tables:
                table_name = table["name"]
                
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_summary = ", ".join([
                    f"{col['name']} ({col['type']}{' NOT NULL' if col['notnull'] else ''}{' PRIMARY KEY' if col['pk'] else ''})".strip() 
                    for col in columns
                ])
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                row_count = cursor.fetchone()["count"]
                
                output.append(f"Table: {table_name}")
                output.append(f"Columns: {column_summary}")
                output.append(f"Row Count: {row_count}")
                output.append("")
            
            return "\n".join(output)
            
    except sqlite3.Error as e:
        logger.error(f"Error in traverse_database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="discover_columns_and_data",
    description="Discover columns and sample data for a specified or selected table as a formatted string."
)
async def discover_columns_and_data(request: DiscoverColumnsRequest) -> str:
    """Discover columns and sample data for a table."""
    global selected_table
    
    table_name = request.table or selected_table
    if not table_name:
        raise HTTPException(
            status_code=400, 
            detail="No table specified or selected. Use edit_table to select a table or provide a table name."
        )
    
    validate_table_exists(table_name)
    
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            output = [f"Table: {table_name}\n"]
            output.append("Column Details:")
            
            for col in columns:
                constraints = []
                if col["notnull"]:
                    constraints.append("NOT NULL")
                if col["pk"]:
                    constraints.append("PRIMARY KEY")
                constraints_str = f" ({', '.join(constraints)})" if constraints else ""
                output.append(f"- {col['name']}: {col['type']}{constraints_str}")
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()
            
            output.append("\nSample Data (up to 5 rows):")
            if not rows:
                output.append("No data found.")
            else:
                headers = [col["name"] for col in columns]
                output.append("| " + " | ".join(headers) + " |")
                output.append("| " + " | ".join(["---" for _ in headers]) + " |")
                for row in rows:
                    row_values = [str(row[col]) if row[col] is not None else "NULL" for col in headers]
                    output.append("| " + " | ".join(row_values) + " |")
            
            return "\n".join(output)
            
    except sqlite3.Error as e:
        logger.error(f"Error in discover_columns_and_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="get_filtered_data",
    description="Retrieve filtered data from a specified or selected table as a formatted string."
)
async def get_filtered_data(request: FilteredDataRequest) -> str:
    """Get filtered data from a table."""
    global selected_table
    
    table_name = request.table or selected_table
    if not table_name:
        raise HTTPException(
            status_code=400, 
            detail="No table specified or selected. Use edit_table to select a table or provide a table name."
        )
    
    validate_table_exists(table_name)
    
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            
            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            
            # Add filters
            if request.filters:
                for condition in request.filters:
                    query += f" AND {condition.column} = ?"
                    params.append(condition.value)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return f"No data found in table '{table_name}' matching the specified criteria.\n"
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            headers = [col["name"] for col in columns]
            
            output = [f"Filtered Data from Table: {table_name}\n"]
            for i, row in enumerate(rows, 1):
                output.append(f"Record {i}:")
                for header in headers:
                    value = row[header] if row[header] is not None else "N/A"
                    output.append(f"  {header}: {value}")
                output.append("")
            
            return "\n".join(output)
            
    except sqlite3.Error as e:
        logger.error(f"Error in get_filtered_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="clear_selected_table",
    description="Clear the currently selected table from the cache."
)
async def clear_selected_table() -> Dict[str, str]:
    """Clear the selected table from cache."""
    global selected_table
    
    if not selected_table:
        return {"message": "No table is currently selected."}
    
    table_name = selected_table
    selected_table = None
    logger.info(f"Cleared selected table '{table_name}' from cache")
    return {"message": f"Cleared selected table '{table_name}' from cache."}


@mcp.tool(
    name="drop_table",
    description="Drop a table from the database."
)
async def drop_table(request: DropTableRequest) -> Dict[str, str]:
    """Drop a table from the database."""
    global selected_table
    
    if not selected_table:
        raise HTTPException(
            status_code=400, 
            detail="No table is currently selected. Use edit_table to select a table first."
        )
    
    validate_table_exists(selected_table)
    
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {selected_table}")
            conn.commit()
            
            table_name = selected_table
            selected_table = None
            
            logger.info(f"Dropped table '{table_name}'")
            return {"message": f"Table '{table_name}' dropped successfully."}
            
    except sqlite3.Error as e:
        logger.error(f"Error in drop_table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.tool(
    name="execute_sql_query",
    description="Execute an arbitrary SQL query on the database and return the result or error. Use with caution!"
)
async def execute_sql_query(sql: str) -> Dict[str, Any]:
    """Execute an arbitrary SQL query."""
    try:
        with safe_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            
            if sql.strip().lower().startswith("select"):
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
                return {"result": result}
            else:
                conn.commit()
                affected = cursor.rowcount
                return {"message": f"Query executed successfully. Rows affected: {affected}"}
                
    except sqlite3.Error as e:
        logger.error(f"Error in execute_sql_query: {e}")
        return {"error": str(e)}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Start the MCP server
    logger.info("Starting Mosaic Database Server...")
    mcp.run(transport="sse")
