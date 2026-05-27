import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from financial_agent.ratio_engine import calculate_ratios
from financial_agent.canonical_map import canonicalize_item
from financial_agent.trend_engine import TrendEngine
from financial_agent.vector_store import FinancialVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "financial_db" / "financial_statements.db"

load_dotenv()


class FinancialAgent:
    def __init__(self, db_path=DB_PATH, model="gpt-4o-mini"):
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                "financial_statements.db not found. Run build_financial_database.py first."
            )

        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

        csv_path = self.db_path.parent / "financial_records.csv"
        self.df = pd.read_csv(csv_path)

        self.df["company"] = self.df["company"].astype(str).str.upper().str.strip()
        self.df["item"] = self.df["item"].astype(str).str.strip()

        self.trend_engine = TrendEngine(self.df)

        try:
            self.vector_store = FinancialVectorStore()
        except Exception:
            self.vector_store = None

        self.dynamic_item_cache = {}

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_company_year_data(self, company: str, year: int, report_type: Optional[str] = None) -> Dict[str, float]:
        company = company.upper().strip()
        params = [company, int(year)]

        report_filter = ""
        if report_type:
            report_filter = " AND r.report_type = ?"
            params.append(report_type)

        query = f"""
        SELECT li.canonical_item AS item, fv.value, s.statement_type, fv.row_number
        FROM financial_values fv
        JOIN line_items li ON li.line_item_id = fv.line_item_id
        JOIN statements s ON s.statement_id = fv.statement_id
        JOIN reports r ON r.report_id = s.report_id
        JOIN companies c ON c.company_id = r.company_id
        WHERE c.ticker = ? AND fv.fiscal_year = ? {report_filter}
        ORDER BY
            CASE s.statement_type
                WHEN 'Income Statement' THEN 1
                WHEN 'Balance Sheet' THEN 2
                WHEN 'Cash Flow' THEN 3
                ELSE 4
            END,
            fv.row_number
        """

        values = {}

        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()

        for row in rows:
            item = canonicalize_item(row["item"])
            if item not in values:
                values[item] = row["value"]

        return values

    def get_trend(self, company: str, item: str, report_type: Optional[str] = None):
        company = company.upper().strip()
        params = [company, item.lower()]

        report_filter = ""
        if report_type:
            report_filter = " AND r.report_type = ?"
            params.append(report_type)

        query = f"""
        SELECT fv.fiscal_year AS year, AVG(fv.value) AS value
        FROM financial_values fv
        JOIN line_items li ON li.line_item_id = fv.line_item_id
        JOIN statements s ON s.statement_id = fv.statement_id
        JOIN reports r ON r.report_id = s.report_id
        JOIN companies c ON c.company_id = r.company_id
        WHERE c.ticker = ? AND lower(li.canonical_item) = ? {report_filter}
        GROUP BY fv.fiscal_year
        ORDER BY fv.fiscal_year
        """

        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return {
            "company": company,
            "metric": item,
            "data": [dict(r) for r in rows]
        }

    def list_companies(self):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT ticker, company_name FROM companies ORDER BY ticker"
            ).fetchall()

        return [dict(r) for r in rows]

    def analyze_trend(self, company, item, statement_type=None, report_type=None):
        trend = self.trend_engine.get_item_trend(
            company=company,
            item=item,
            statement_type=statement_type,
            report_type=report_type
        )

        summary = self.trend_engine.summarize_trend(
            trend,
            item.replace("_", " ").title()
        )

        return {
            "company": company.upper(),
            "metric": item,
            "statement_type": statement_type,
            "report_type": report_type,
            "trend": trend,
            "summary": summary
        }

    def analyze_company_performance_trends(self, company, report_type="Audited Report"):
        metrics = [
            ("Revenue", "Income Statement"),
            ("Net Profit", "Income Statement"),
            ("Total Assets", "Balance Sheet"),
            ("Total Liabilities", "Balance Sheet"),
            ("Total Equity", "Balance Sheet"),
            ("Net Cash from Operating Activities", "Cash Flow"),
        ]

        results = {}

        for item, statement_type in metrics:
            trend = self.trend_engine.get_item_trend(
                company=company,
                item=item,
                statement_type=statement_type,
                report_type=report_type
            )

            summary = self.trend_engine.summarize_trend(trend, item)

            results[item] = {
                "statement_type": statement_type,
                "trend": trend,
                "summary": summary
            }

        return {
            "company": company.upper(),
            "report_type": report_type,
            "results": results
        }

    def get_financial_value(
        self,
        company,
        item,
        year=None,
        report_type=None,
        statement_type=None,
        report_year=None
    ):
        company = company.upper().strip()
        item = str(item).strip()

        data = self.df[
            (self.df["company"] == company) &
            (self.df["item"].str.lower() == item.lower())
        ]

        if year:
            data = data[data["year"] == int(year)]

        if report_type:
            data = data[data["report_type"] == report_type]

        if statement_type:
            data = data[data["statement_type"] == statement_type]

        if report_year:
            data = data[data["report_year"] == int(report_year)]

        if "column_name" in data.columns:
            data = data[
                ~data["column_name"].astype(str).str.lower().fillna("").isin(
                    ["sl. no.", "sl no", "s.no", "s no", "no.", "no"]
                )
            ]

        if data.empty:
            return {
                "company": company,
                "item": item,
                "year": year,
                "report_type": report_type,
                "statement_type": statement_type,
                "found": False,
                "message": "No matching value found."
            }

        columns = [
            "company",
            "report_type",
            "report_year",
            "statement_type",
            "year",
            "source_file",
            "sheet_name",
            "column_name",
            "raw_item",
            "item",
            "value"
        ]

        available_columns = [col for col in columns if col in data.columns]

        results = data[available_columns].drop_duplicates().to_dict(orient="records")

        return {
            "company": company,
            "item": item,
            "year": year,
            "report_type": report_type,
            "statement_type": statement_type,
            "found": True,
            "results": results
        }

    def safe_generate_explanation(self, question, result):
        try:
            return self.generate_financial_explanation(question, result)
        except Exception as e:
            return f"AI explanation unavailable: {str(e)}"

    def answer_user_question(self, question):
        from financial_agent.question_router import (
            detect_company,
            detect_year,
            detect_report_type,
            detect_item,
            detect_statement_type,
            detect_intent,
        )

        companies = sorted(self.df["company"].dropna().unique())
        company = detect_company(question, companies)
        year = detect_year(question)
        report_type = detect_report_type(question)
        intent = detect_intent(question)

        if company:
            all_valid_items = list(
                self.df[self.df["company"] == company]["item"]
                .dropna()
                .astype(str)
                .unique()
            )
        else:
            all_valid_items = list(
                self.df["item"]
                .dropna()
                .astype(str)
                .unique()
            )

        cache_key = f"{company}_{question.lower().strip()}"

        if cache_key in self.dynamic_item_cache:
            item = self.dynamic_item_cache[cache_key]
        else:
            item = detect_item(question, valid_items_from_db=all_valid_items)
            if item and company:
                self.dynamic_item_cache[cache_key] = item

        if not company:
            return {
                "type": "clarification",
                "answer": "Please mention the company ticker, for example BBPL, BPCL, BNBL, or TBANK."
            }

        if intent == "summary":
            result = self.analyze_company_performance_trends(
                company=company,
                report_type=report_type or "Audited Report"
            )

            lines = [f"Financial performance trend for {company}:"]

            for metric, data in result["results"].items():
                lines.append(f"- {metric}: {data['summary']}")

            summary_answer = "\n".join(lines)

            explanation = self.safe_generate_explanation(question, result)

            return {
                "type": "summary",
                "company": company,
                "report_type": report_type or "Audited Report",
                "answer": summary_answer,
                "explanation": explanation,
                "data": result
            }

        if intent == "trend":
            if not item:
                return {
                    "type": "clarification",
                    "company": company,
                    "answer": "Please mention the financial metric, for example revenue, net profit, total assets, or operating cash flow."
                }

            statement_type = detect_statement_type(item)

            result = self.analyze_trend(
                company=company,
                item=item,
                statement_type=statement_type,
                report_type=report_type or "Audited Report"
            )

            explanation = self.safe_generate_explanation(question, result)

            return {
                "type": "trend",
                "company": company,
                "item": item,
                "answer": result["summary"],
                "explanation": explanation,
                "trend_data": result["trend"],
                "data": result
            }

        if not item:
            return {
                "type": "clarification",
                "company": company,
                "answer": "Please mention what financial item you want, for example EPS, revenue, gross profit, cost of sales, or total assets."
            }

        statement_type = detect_statement_type(item)

        result = self.get_financial_value(
            company=company,
            item=item,
            year=year,
            report_year=year,
            report_type=report_type,
            statement_type=statement_type
        )

        if not result["found"]:
            result = self.get_financial_value(
                company=company,
                item=item,
                year=year,
                report_year=year,
                report_type=report_type,
                statement_type=None
            )

        if not result["found"]:
            return {
                "type": "lookup",
                "company": company,
                "item": item,
                "answer": f"I could not find {item} for {company}."
            }

        rows = result["results"]
        best = rows[0]
        value = best.get("value")

        try:
            float_val = float(value)
            if float_val.is_integer():
                formatted_value = f"{int(float_val):,}"
            else:
                formatted_value = f"{float_val:,.2f}"
        except Exception:
            formatted_value = str(value)

        answer = (
            f"{item} for {company}"
            f"{' in ' + str(year) if year else ''}"
            f"{' (' + report_type + ')' if report_type else ''}"
            f" is {formatted_value}."
        )

        explanation = self.safe_generate_explanation(question, result)

        return {
            "type": "lookup",
            "company": company,
            "item": item,
            "answer": answer,
            "explanation": explanation,
            "data": result
        }

    def answer_question(
        self,
        question: str,
        company: str,
        year: Optional[int] = None,
        report_type: Optional[str] = None
    ) -> Dict[str, Any]:
        company = company.upper().strip()
        values = self.get_company_year_data(company, year, report_type) if year else {}
        ratios = calculate_ratios(values) if values else {}

        if not self.client:
            return {
                "company": company,
                "year": year,
                "report_type": report_type,
                "values": values,
                "ratios": ratios,
                "answer": "OPENAI_API_KEY is not set. The database query worked, but AI explanation is disabled."
            }

        prompt = f"""
You are a professional financial statement analyst for Bhutanese listed companies.

Question: {question}
Company: {company}
Year: {year}
Report type: {report_type or 'Any'}

Retrieved exact financial values:
{values}

Calculated ratios:
{ratios}

Rules:
- Use only the retrieved values and calculated ratios above.
- If a value is missing, say it is not available in the database.
- Do not invent numbers.
- For profitability, use revenue, gross profit, net profit, ROA, ROE, and margin where available.
- For liquidity, use current assets, current liabilities, current ratio, and cash where available.
- For solvency, use liabilities, equity, debt ratio, and debt-to-equity where available.
- For cash flow, use net cash from operating activities and cash balances where available.
- Keep the answer clear, professional, and concise.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful financial analyst. Never hallucinate financial figures."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=450
            )

            answer = response.choices[0].message.content

        except Exception as e:
            answer = f"AI explanation unavailable: {str(e)}"

        return {
            "company": company,
            "year": year,
            "report_type": report_type,
            "values": values,
            "ratios": ratios,
            "answer": answer
        }

    def generate_financial_explanation(self, question, data):
        if not self.client:
            return "OpenAI client is not configured."

        try:
            rag_context = ""

            try:
                company = data.get("company")
                year = data.get("year")
                report_type = data.get("report_type")

                if self.vector_store and company:
                    rag_results = self.vector_store.search(
                        question=question,
                        company=company,
                        report_type=report_type,
                        report_year=year,
                        year=year,
                        n_results=5
                    )

                    rag_docs = rag_results.get("documents", [[]])[0]

                    if rag_docs:
                        rag_context = "\n".join([str(doc) for doc in rag_docs])

            except Exception as e:
                rag_context = f"RAG context unavailable: {e}"

            prompt = f"""
You are a professional financial analyst for Bhutanese listed companies.

User Question:
{question}

Structured Financial Data:
{data}

Semantic RAG Context:
{rag_context}

Instructions:
- Use only the provided structured data and RAG context.
- Do not invent company names, values, or years.
- Clearly mention whether the data is Audited Report or Interim Report when available.
- Explain what the figure or trend means for profitability, liquidity, solvency, or growth when relevant.
- If the value is from Operational Schedule, explain that it may be a supporting/segment-level value, not a full statutory statement figure.
- Keep the response concise and professional.
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional financial analyst. Use only the provided data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=600
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"AI explanation unavailable: {str(e)}"