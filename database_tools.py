"""
Database Tools - Consolidated tools for FFIEC table with flexible filtering
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class FFIECTools:
    """Consolidated tools for FFIEC banking data - fewer, more powerful tools"""

    def __init__(self, engine: Engine):
        """Initialize FFIEC tools with database engine"""
        self.engine = engine
        self.table_name = "ffiec"
        logger.info("Initialized FFIECTools")

    def create_tools(self) -> List:
        """Create consolidated FFIEC tools"""
        tools = []

        # Tool 1: Query FFIEC Data (handles most queries)
        @tool
        def query_ffiec_data(
            bank_identifier: Optional[str] = None,
            identifier_type: str = "name",
            reporting_date: Optional[str] = None,
            report_elements: Optional[List[str]] = None,
            schedules: Optional[List[str]] = None,
            source_item_pattern: Optional[str] = None,
            columns: Optional[List[str]] = None,
            limit: int = 100,
        ) -> str:
            """
            Query FFIEC banking data with flexible filtering.

            This tool handles most FFIEC queries including:
            - Total assets by bank and date
            - Capital structure queries
            - Balance sheet items (Schedule RC)
            - Risk-weighted assets (Schedule RC-R)
            - Specific report elements or patterns
            - Cross-schedule comparisons

            The LLM should extract appropriate filters from user queries.

            Args:
                bank_identifier: Bank Name (e.g., "Metro Financial Corporation") or Bank ID (e.g., "BANK001")
                                 Leave None to query all banks
                identifier_type: "name" or "id" (default: "name")
                reporting_date: Date in YYYY-MM-DD format (e.g., "2024-12-31"). None for all dates
                report_elements: List of report element categories to filter by
                                 Examples: ["TOTAL ASSETS"], ["TOTAL ASSETS", "CAPITAL"]
                                 Use None for all report elements
                schedules: List of schedule codes to filter by
                          Examples: ["RC"], ["RC", "RC-R"], ["RC-G"]
                          None for all schedules
                source_item_pattern: Pattern to match source items (supports SQL LIKE)
                                    Examples: "%Cash%", "%Tier 1%", "%loan%"
                                    None for no filtering
                columns: Specific columns to return. None for all columns
                        Examples: ["Bank Name", "Value"], ["Source Item", "Value", "Reporting Date"]
                limit: Maximum rows to return (default: 100)

            Returns:
                JSON string with query results

            Example Use Cases:

            1. Total assets for a bank:
               query_ffiec_data(
                   bank_identifier="Metro Financial Corporation",
                   reporting_date="2024-12-31",
                   report_elements=["TOTAL ASSETS"]
               )

            2. Capital structure:
               query_ffiec_data(
                   bank_identifier="BANK001",
                   identifier_type="id",
                   reporting_date="2024-12-31",
                   source_item_pattern="%Common stock%|%Surplus%|%Retained earnings%"
               )

            3. Balance sheet items (Schedule RC):
               query_ffiec_data(
                   bank_identifier="Metro Financial Corporation",
                   reporting_date="2024-12-31",
                   schedules=["RC"]
               )

            4. Risk-weighted assets (Schedule RC-R):
               query_ffiec_data(
                   bank_identifier="BANK001",
                   identifier_type="id",
                   reporting_date="2024-12-31",
                   schedules=["RC-R"]
               )

            5. Compare schedules:
               query_ffiec_data(
                   bank_identifier="BANK004",
                   identifier_type="id",
                   reporting_date="2024-12-31",
                   schedules=["RC", "RC-R"]
               )

            6. All loan-related items:
               query_ffiec_data(
                   bank_identifier="Metro Financial Corporation",
                   source_item_pattern="%loan%"
               )

            7. Banks with assets over threshold (use calculate_and_filter):
               First query all banks' assets, then filter in LLM
            """
            return self._query_ffiec_data(
                bank_identifier,
                identifier_type,
                reporting_date,
                report_elements,
                schedules,
                source_item_pattern,
                columns,
                limit,
            )

        # Tool 2: Calculate and Aggregate (handles calculations, ratios, aggregations)
        @tool
        def calculate_and_aggregate(
            bank_identifier: Optional[str] = None,
            identifier_type: str = "name",
            reporting_date: Optional[str] = None,
            calculation_type: str = "sum",
            group_by: Optional[List[str]] = None,
            report_elements: Optional[List[str]] = None,
            schedules: Optional[List[str]] = None,
            source_item_pattern: Optional[str] = None,
        ) -> str:
            """
            Calculate aggregations, ratios, and perform calculations on FFIEC data.

            Use this for:
            - Summing/averaging values
            - Calculating capital ratios
            - Grouping data by categories
            - Computing totals and subtotals

            Args:
                bank_identifier: Bank Name or Bank ID (None for all banks)
                identifier_type: "name" or "id"
                reporting_date: Date in YYYY-MM-DD format (None for all dates)
                calculation_type: Type of calculation
                                 - "sum": Sum values
                                 - "avg": Average values
                                 - "count": Count records
                                 - "tier1_ratio": Calculate Tier 1 capital ratio
                                 - "total_capital_ratio": Calculate total capital ratio
                group_by: Columns to group by
                         Examples: ["Bank Name"], ["Report Element"], ["Schedule"]
                report_elements: Filter by report elements
                schedules: Filter by schedules
                source_item_pattern: Filter by source item pattern

            Returns:
                JSON string with calculated results

            Example Use Cases:

            1. Sum total assets:
               calculate_and_aggregate(
                   bank_identifier="Metro Financial Corporation",
                   reporting_date="2024-12-31",
                   calculation_type="sum",
                   report_elements=["TOTAL ASSETS"]
               )

            2. Calculate Tier 1 ratio:
               calculate_and_aggregate(
                   bank_identifier="BANK001",
                   identifier_type="id",
                   reporting_date="2024-12-31",
                   calculation_type="tier1_ratio"
               )

            3. Sum by report element:
               calculate_and_aggregate(
                   bank_identifier="BANK001",
                   identifier_type="id",
                   reporting_date="2024-12-31",
                   calculation_type="sum",
                   group_by=["Report Element"]
               )

            4. Average values by schedule:
               calculate_and_aggregate(
                   bank_identifier="Metro Financial Corporation",
                   calculation_type="avg",
                   group_by=["Schedule"]
               )
            """
            return self._calculate_and_aggregate(
                bank_identifier,
                identifier_type,
                reporting_date,
                calculation_type,
                group_by,
                report_elements,
                schedules,
                source_item_pattern,
            )

        # Tool 3: Get Schema Info (for discovery)
        @tool
        def get_ffiec_schema_info(info_type: str = "overview") -> str:
            """
            Get schema information about FFIEC table.

            Use for discovery queries or when user wants to know what data is available.

            Args:
                info_type: Type of information to retrieve
                          - "overview": Table overview with columns
                          - "banks": List of available banks
                          - "dates": Available reporting dates
                          - "report_elements": Available report elements
                          - "schedules": Available schedule codes

            Returns:
                JSON string with schema information
            """
            return self._get_schema_info(info_type)

        tools.extend([query_ffiec_data, calculate_and_aggregate, get_ffiec_schema_info])

        logger.info(f"Created {len(tools)} consolidated FFIEC tools")
        return tools

    # Implementation methods

    def _query_ffiec_data(
        self,
        bank_identifier: Optional[str],
        identifier_type: str,
        reporting_date: Optional[str],
        report_elements: Optional[List[str]],
        schedules: Optional[List[str]],
        source_item_pattern: Optional[str],
        columns: Optional[List[str]],
        limit: int,
    ) -> str:
        """Implementation of query_ffiec_data"""
        try:
            params = {}
            
            # Build SELECT clause
            # Note: Column selection is still string interpolation but controlled by allowed columns
            # Ideally this should be validated against schema, but for now we trust the tool definition
            if columns:
                # Basic sanitization to ensure only alphanumeric and spaces
                safe_columns = [c for c in columns if c.replace(" ", "").isalnum()]
                if not safe_columns:
                    select_clause = "*"
                else:
                    select_clause = ", ".join([f'"{col}"' for col in safe_columns])
            else:
                select_clause = "*"

            # Build query
            query_str = f'SELECT {select_clause} FROM "{self.table_name}" WHERE 1=1'

            # Add filters with parameters
            if bank_identifier:
                column = "Bank Name" if identifier_type == "name" else "Bank ID"
                query_str += f" AND \"{column}\" = :bank_identifier"
                params["bank_identifier"] = bank_identifier

            if reporting_date:
                query_str += " AND \"Reporting Date\" = :reporting_date"
                params["reporting_date"] = reporting_date

            if report_elements:
                query_str += " AND \"Report Element\" IN :report_elements"
                params["report_elements"] = tuple(report_elements)

            if schedules:
                query_str += " AND \"Schedule\" IN :schedules"
                params["schedules"] = tuple(schedules)

            if source_item_pattern:
                # Handle multiple patterns separated by |
                if "|" in source_item_pattern:
                    patterns = source_item_pattern.split("|")
                    pattern_conditions = []
                    for i, p in enumerate(patterns):
                        param_name = f"pattern_{i}"
                        pattern_conditions.append(f"\"Source Item\" LIKE :{param_name}")
                        params[param_name] = p.strip()
                    query_str += f' AND ({" OR ".join(pattern_conditions)})'
                else:
                    query_str += " AND \"Source Item\" LIKE :source_item_pattern"
                    params["source_item_pattern"] = source_item_pattern

            query_str += f' ORDER BY "Bank Name", "Reporting Date" DESC, "Report Element" LIMIT :limit'
            params["limit"] = limit

            logger.info(f"Executing query_ffiec_data: {query_str[:200]}...")

            with self.engine.connect() as conn:
                result = conn.execute(text(query_str), params)
                rows = result.fetchall()
                columns_result = result.keys()

                data = [dict(zip(columns_result, row)) for row in rows]

                logger.info(f"Query returned {len(data)} rows")

                return json.dumps(
                    {"success": True, "count": len(data), "data": data, "query_preview": query_str[:200]}, default=str
                )

        except Exception as e:
            logger.error(f"Error in query_ffiec_data: {e}", exc_info=True)
            return json.dumps({"success": False, "error": str(e)})

    def _calculate_and_aggregate(
        self,
        bank_identifier: Optional[str],
        identifier_type: str,
        reporting_date: Optional[str],
        calculation_type: str,
        group_by: Optional[List[str]],
        report_elements: Optional[List[str]],
        schedules: Optional[List[str]],
        source_item_pattern: Optional[str],
    ) -> str:
        """Implementation of calculate_and_aggregate"""
        try:
            # Special handling for ratio calculations
            if calculation_type in ["tier1_ratio", "total_capital_ratio"]:
                return self._calculate_capital_ratio(
                    bank_identifier, identifier_type, reporting_date, calculation_type
                )

            params = {}
            
            # Build aggregation query
            select_parts = []

            # Add grouping columns
            if group_by:
                # Basic sanitization
                safe_group_by = [c for c in group_by if c.replace(" ", "").isalnum()]
                select_parts.extend([f'"{col}"' for col in safe_group_by])

            # Add aggregation
            if calculation_type == "sum":
                select_parts.append('SUM("Value") as "Total"')
            elif calculation_type == "avg":
                select_parts.append('AVG("Value") as "Average"')
            elif calculation_type == "count":
                select_parts.append('COUNT(*) as "Count"')
            else:
                select_parts.append('SUM("Value") as "Total"')

            select_clause = ", ".join(select_parts)
            query_str = f'SELECT {select_clause} FROM "{self.table_name}" WHERE 1=1'

            # Add filters
            if bank_identifier:
                column = "Bank Name" if identifier_type == "name" else "Bank ID"
                query_str += f" AND \"{column}\" = :bank_identifier"
                params["bank_identifier"] = bank_identifier

            if reporting_date:
                query_str += " AND \"Reporting Date\" = :reporting_date"
                params["reporting_date"] = reporting_date

            if report_elements:
                query_str += " AND \"Report Element\" IN :report_elements"
                params["report_elements"] = tuple(report_elements)

            if schedules:
                query_str += " AND \"Schedule\" IN :schedules"
                params["schedules"] = tuple(schedules)

            if source_item_pattern:
                query_str += " AND \"Source Item\" LIKE :source_item_pattern"
                params["source_item_pattern"] = source_item_pattern

            # Add GROUP BY
            if group_by:
                safe_group_by = [c for c in group_by if c.replace(" ", "").isalnum()]
                if safe_group_by:
                    group_clause = ", ".join([f'"{col}"' for col in safe_group_by])
                    query_str += f" GROUP BY {group_clause}"

            query_str += " LIMIT 100"

            logger.info(f"Executing calculate_and_aggregate: {query_str[:200]}...")

            with self.engine.connect() as conn:
                result = conn.execute(text(query_str), params)
                rows = result.fetchall()
                columns_result = result.keys()

                data = [dict(zip(columns_result, row)) for row in rows]

                logger.info(f"Aggregation returned {len(data)} rows")

                return json.dumps(
                    {"success": True, "calculation_type": calculation_type, "count": len(data), "data": data},
                    default=str,
                )

        except Exception as e:
            logger.error(f"Error in calculate_and_aggregate: {e}", exc_info=True)
            return json.dumps({"success": False, "error": str(e)})

    def _calculate_capital_ratio(
        self, bank_identifier: Optional[str], identifier_type: str, reporting_date: Optional[str], ratio_type: str
    ) -> str:
        """Calculate capital ratios"""
        try:
            column = "Bank Name" if identifier_type == "name" else "Bank ID"
            params = {
                "bank_identifier": bank_identifier,
                "reporting_date": reporting_date
            }

            # Get relevant data for ratio calculation
            query_str = f"""
            SELECT "Source Item", "Value"
            FROM "{self.table_name}"
            WHERE "{column}" = :bank_identifier
            AND "Reporting Date" = :reporting_date
            AND ("Source Item" LIKE '%Tier 1%' OR "Source Item" LIKE '%risk-weighted%'
                 OR "Source Item" LIKE '%Capital%')
            """

            logger.info(f"Calculating {ratio_type} for {bank_identifier}")

            with self.engine.connect() as conn:
                result = conn.execute(text(query_str), params)
                rows = result.fetchall()

                # Extract values
                tier1_capital = None
                rwa = None
                reported_ratio = None

                for row in rows:
                    source_item = row[0].lower() if row[0] else ""
                    value = row[1]

                    if "tier 1 capital" in source_item and "ratio" not in source_item:
                        tier1_capital = value
                    elif "gross risk-weighted assets" in source_item:
                        rwa = value
                    elif "tier 1" in source_item and "ratio" in source_item:
                        reported_ratio = value

                # Calculate
                calculated_ratio = None
                if tier1_capital and rwa and rwa > 0:
                    calculated_ratio = (tier1_capital / rwa) * 100

                verification = "INCOMPLETE_DATA"
                if calculated_ratio and reported_ratio:
                    verification = "MATCH" if abs(calculated_ratio - reported_ratio) < 0.01 else "MISMATCH"

                return json.dumps(
                    {
                        "success": True,
                        "bank": bank_identifier,
                        "date": reporting_date,
                        "ratio_type": ratio_type,
                        "components": {"tier1_capital": tier1_capital, "risk_weighted_assets": rwa},
                        "calculated_ratio": round(calculated_ratio, 2) if calculated_ratio else None,
                        "reported_ratio": reported_ratio,
                        "verification": verification,
                        "formula": "(Tier 1 Capital / Risk-Weighted Assets) × 100",
                    },
                    default=str,
                )

        except Exception as e:
            logger.error(f"Error calculating capital ratio: {e}", exc_info=True)
            return json.dumps({"success": False, "error": str(e)})

    def _get_schema_info(self, info_type: str) -> str:
        """Get schema information"""
        try:
            if info_type == "banks":
                query_str = 'SELECT DISTINCT "Bank ID", "Bank Name" FROM ffiec ORDER BY "Bank Name" LIMIT 50'
            elif info_type == "dates":
                query_str = 'SELECT DISTINCT "Reporting Date" FROM ffiec ORDER BY "Reporting Date" DESC LIMIT 20'
            elif info_type == "report_elements":
                query_str = 'SELECT DISTINCT "Report Element" FROM ffiec ORDER BY "Report Element" LIMIT 50'
            elif info_type == "schedules":
                query_str = 'SELECT DISTINCT "Schedule" FROM ffiec ORDER BY "Schedule"'
            else:  # overview
                return json.dumps(
                    {
                        "success": True,
                        "table": "ffiec",
                        "columns": {
                            "Bank ID": "varchar(100)",
                            "Bank Name": "varchar(100)",
                            "Report Element": "varchar(100)",
                            "Item Number": "varchar(100)",
                            "Schedule": "varchar(100)",
                            "Source Item": "varchar(500)",
                            "Source Item Number": "varchar(500)",
                            "Value": "bigint",
                            "Reporting Date": "date",
                        },
                        "description": "FFIEC banking regulatory data",
                    }
                )

            with self.engine.connect() as conn:
                result = conn.execute(text(query_str))
                rows = result.fetchall()
                columns = result.keys()

                data = [dict(zip(columns, row)) for row in rows]

                return json.dumps({"success": True, "info_type": info_type, "data": data}, default=str)

        except Exception as e:
            logger.error(f"Error getting schema info: {e}", exc_info=True)
            return json.dumps({"success": False, "error": str(e)})


def create_database_tools(engine: Engine) -> List:
    """Create all table-specific database tools"""
    tools = []

    # Create FFIEC tools (now only 3 tools)
    ffiec_tools = FFIECTools(engine)
    tools.extend(ffiec_tools.create_tools())

    logger.info(f"Created {len(tools)} database tools total")
    return tools
