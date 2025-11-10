import json
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class FFIECDatabaseTool:
    """
    Table-specific tool for FFIEC banking data

    FFIEC Table Schema:
    - Bank ID: Unique bank identifier (e.g., 'BANK001')
    - Bank Name: Financial institution name (e.g., 'Metro Financial Corporation')
    - Report Element: Category (e.g., 'TOTAL ASSETS', 'GENERAL LOAN AND LEASE VALUATION ALLOWANCES')
    - Item Number: Report item number
    - Schedule: Reporting schedule code
    - Source Item: Detailed line item description
    - Source Item Number: Source identifier
    - Value: Numeric value (in dollars)
    - Reporting Date: Report date (YYYY-MM-DD)
    """

    def __init__(self, engine: Engine):
        """Initialize FFIEC tool with database engine"""
        self.engine = engine
        self.table_name = "ffiec"
        logger.info("Initialized FFIECDatabaseTool")

    def create_tool(self):
        """Create the FFIEC query tool"""

        @tool
        def query_ffiec(
            select_columns: Optional[List[str]] = None,
            where_conditions: Optional[List[Dict[str, Any]]] = None,
            group_by_columns: Optional[List[str]] = None,
            aggregations: Optional[List[Dict[str, str]]] = None,
            order_by: Optional[List[Dict[str, str]]] = None,
            limit: int = 100,
        ) -> str:
            """
            Query the FFIEC banking data table.

            This tool queries FFIEC banking regulatory data including assets, liabilities,
            loans, and other financial metrics reported by banks.

            The LLM should analyze the user's natural language query and extract:
            - Which columns to retrieve
            - What filters to apply
            - Any aggregations needed
            - Grouping and sorting requirements

            Available Columns:
            - "Bank ID": Unique identifier (e.g., 'BANK001')
            - "Bank Name": Institution name (e.g., 'Metro Financial Corporation')
            - "Report Element": Category (e.g., 'TOTAL ASSETS')
            - "Item Number": Report item number
            - "Schedule": Schedule code
            - "Source Item": Detailed line item (e.g., 'Cash and balances due from depository institutions')
            - "Source Item Number": Source ID
            - "Value": Numeric amount in dollars
            - "Reporting Date": Date (YYYY-MM-DD format)

            Args:
                select_columns: List of columns to return. Examples:
                    - ["Bank Name", "Value", "Reporting Date"]
                    - ["Report Element", "Source Item"]
                    - None or ["*"] for all columns

                where_conditions: List of filter conditions. Each condition is a dict:
                    {
                        "column": "Bank Name",
                        "operator": "=",  # =, !=, >, <, >=, <=, LIKE, IN, BETWEEN
                        "value": "Metro Financial Corporation",
                        "logical": "AND"  # AND, OR (default: AND)
                    }
                    Examples:
                    - [{"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"}]
                    - [{"column": "Value", "operator": ">", "value": 1000000000}]
                    - [{"column": "Report Element", "operator": "=", "value": "TOTAL ASSETS"}]
                    - [{"column": "Reporting Date", "operator": "=", "value": "2024-12-31"}]
                    - [{"column": "Source Item", "operator": "LIKE", "value": "%Cash%"}]

                group_by_columns: Columns to group by. Examples:
                    - ["Bank Name"]
                    - ["Report Element", "Reporting Date"]

                aggregations: Aggregate functions to apply. Each is a dict:
                    {
                        "function": "SUM",  # SUM, AVG, COUNT, MIN, MAX
                        "column": "Value",
                        "alias": "Total Value"
                    }
                    Examples:
                    - [{"function": "SUM", "column": "Value", "alias": "Total"}]
                    - [{"function": "COUNT", "column": "*", "alias": "Count"}]
                    - [{"function": "AVG", "column": "Value", "alias": "Average"}]

                order_by: Sort specification. Each is a dict:
                    {
                        "column": "Value",
                        "direction": "DESC"  # ASC or DESC
                    }
                    Examples:
                    - [{"column": "Value", "direction": "DESC"}]
                    - [{"column": "Bank Name", "direction": "ASC"}]

                limit: Maximum rows to return (default: 100, max: 1000)

            Returns:
                JSON string with query results:
                {
                    "success": true,
                    "table": "ffiec",
                    "count": 10,
                    "data": [{"Bank Name": "...", "Value": ...}, ...]
                }

            Example Usage Scenarios:

            1. Simple query - "Show me Metro Financial's total assets"
               query_ffiec(
                   select_columns=["Bank Name", "Report Element", "Value"],
                   where_conditions=[
                       {"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"},
                       {"column": "Report Element", "operator": "=", "value": "TOTAL ASSETS"}
                   ]
               )

            2. Aggregation - "Sum all values by bank name"
               query_ffiec(
                   select_columns=["Bank Name"],
                   group_by_columns=["Bank Name"],
                   aggregations=[{"function": "SUM", "column": "Value", "alias": "Total"}]
               )

            3. Filtered aggregation - "Average asset values for Metro Financial"
               query_ffiec(
                   select_columns=["Report Element"],
                   where_conditions=[
                       {"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"},
                       {"column": "Report Element", "operator": "=", "value": "TOTAL ASSETS"}
                   ],
                   group_by_columns=["Report Element"],
                   aggregations=[{"function": "AVG", "column": "Value", "alias": "Average"}]
               )

            4. Time series - "Show Metro Financial's cash balances over time"
               query_ffiec(
                   select_columns=["Reporting Date", "Value"],
                   where_conditions=[
                       {"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"},
                       {"column": "Source Item", "operator": "LIKE", "value": "%Cash%"}
                   ],
                   order_by=[{"column": "Reporting Date", "direction": "ASC"}]
               )

            5. Top N query - "Top 5 largest asset items for Metro Financial"
               query_ffiec(
                   select_columns=["Source Item", "Value"],
                   where_conditions=[
                       {"column": "Bank Name", "operator": "=", "value": "Metro Financial Corporation"},
                       {"column": "Report Element", "operator": "=", "value": "TOTAL ASSETS"}
                   ],
                   order_by=[{"column": "Value", "direction": "DESC"}],
                   limit=5
               )
            """
            return self._execute_ffiec_query(
                select_columns=select_columns,
                where_conditions=where_conditions,
                group_by_columns=group_by_columns,
                aggregations=aggregations,
                order_by=order_by,
                limit=limit,
            )

        logger.info("Created query_ffiec tool")
        return query_ffiec

    def _execute_ffiec_query(
        self,
        select_columns: Optional[List[str]] = None,
        where_conditions: Optional[List[Dict[str, Any]]] = None,
        group_by_columns: Optional[List[str]] = None,
        aggregations: Optional[List[Dict[str, str]]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        limit: int = 100,
    ) -> str:
        """Execute FFIEC query with given parameters"""

        try:
            # Cap limit
            limit = min(limit, 1000)

            # Build SELECT clause
            select_clause = self._build_select_clause(select_columns, aggregations)

            # Build WHERE clause
            where_clause = self._build_where_clause(where_conditions)

            # Build GROUP BY clause
            group_by_clause = self._build_group_by_clause(group_by_columns)

            # Build ORDER BY clause
            order_by_clause = self._build_order_by_clause(order_by)

            # Construct full query
            query = f'SELECT {select_clause} FROM "{self.table_name}"'

            if where_clause:
                query += f" WHERE {where_clause}"

            if group_by_clause:
                query += f" GROUP BY {group_by_clause}"

            if order_by_clause:
                query += f" ORDER BY {order_by_clause}"

            query += f" LIMIT {limit}"

            logger.info(f"Executing FFIEC query: {query[:200]}...")
            logger.debug(f"Full query: {query}")

            # Execute query
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                columns_result = result.keys()

                data = [dict(zip(columns_result, row)) for row in rows]

                logger.info(f"FFIEC query returned {len(data)} rows")

                return json.dumps(
                    {"success": True, "table": self.table_name, "query": query, "count": len(data), "data": data},
                    default=str,
                )

        except Exception as e:
            logger.error(f"Error executing FFIEC query: {str(e)}", exc_info=True)
            return json.dumps(
                {
                    "success": False,
                    "table": self.table_name,
                    "error": str(e),
                    "query": query if "query" in locals() else None,
                }
            )

    def _build_select_clause(
        self, select_columns: Optional[List[str]] = None, aggregations: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build SELECT clause"""
        parts = []

        # Add regular columns
        if select_columns and select_columns != ["*"]:
            for col in select_columns:
                parts.append(f'"{col}"')
        elif not aggregations:
            # Default to all columns if no specific selection and no aggregations
            parts.append("*")
        elif select_columns:
            # Include selected columns even with aggregations (for GROUP BY)
            for col in select_columns:
                if col != "*":
                    parts.append(f'"{col}"')

        # Add aggregations
        if aggregations:
            for agg in aggregations:
                func = agg.get("function", "COUNT").upper()
                col = agg.get("column", "*")
                alias = agg.get("alias", f"{func}_{col}")

                if col == "*":
                    agg_expr = f"{func}(*)"
                else:
                    agg_expr = f'{func}("{col}")'

                parts.append(f'{agg_expr} AS "{alias}"')

        return ", ".join(parts) if parts else "*"

    def _build_where_clause(self, where_conditions: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build WHERE clause"""
        if not where_conditions:
            return ""

        conditions = []
        for i, cond in enumerate(where_conditions):
            column = cond.get("column")
            operator = cond.get("operator", "=").upper()
            value = cond.get("value")
            logical = cond.get("logical", "AND").upper() if i > 0 else ""

            # Build condition
            if operator in ["IS NULL", "IS NOT NULL"]:
                condition = f'"{column}" {operator}'
            elif operator == "IN":
                if isinstance(value, list):
                    values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                    condition = f'"{column}" IN ({values_str})'
                else:
                    condition = f'"{column}" IN ({value})'
            elif operator == "BETWEEN":
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    val1 = f"'{value[0]}'" if isinstance(value[0], str) else str(value[0])
                    val2 = f"'{value[1]}'" if isinstance(value[1], str) else str(value[1])
                    condition = f'"{column}" BETWEEN {val1} AND {val2}'
                else:
                    raise ValueError("BETWEEN requires list/tuple with 2 values")
            elif operator == "LIKE":
                condition = f"\"{column}\" LIKE '{value}'"
            else:
                if isinstance(value, str):
                    condition = f"\"{column}\" {operator} '{value}'"
                else:
                    condition = f'"{column}" {operator} {value}'

            # Add with logical connector
            if logical and i > 0:
                conditions.append(f" {logical} {condition}")
            else:
                conditions.append(condition)

        return "".join(conditions)

    def _build_group_by_clause(self, group_by_columns: Optional[List[str]] = None) -> str:
        """Build GROUP BY clause"""
        if not group_by_columns:
            return ""

        return ", ".join([f'"{col}"' for col in group_by_columns])

    def _build_order_by_clause(self, order_by: Optional[List[Dict[str, str]]] = None) -> str:
        """Build ORDER BY clause"""
        if not order_by:
            return ""

        parts = []
        for order in order_by:
            column = order.get("column")
            direction = order.get("direction", "ASC").upper()
            parts.append(f'"{column}" {direction}')

        return ", ".join(parts)


# Factory function to create all table-specific tools
def create_database_tools(engine: Engine) -> List:
    """
    Create all table-specific database tools

    Args:
        engine: SQLAlchemy engine

    Returns:
        List of tool instances
    """
    tools = []

    # Create FFIEC tool
    ffiec_tool = FFIECDatabaseTool(engine)
    tools.append(ffiec_tool.create_tool())

    # Add more table-specific tools here as needed
    # Example:
    # customers_tool = CustomersDatabaseTool(engine)
    # tools.append(customers_tool.create_tool())

    logger.info(f"Created {len(tools)} table-specific database tools")
    return tools
