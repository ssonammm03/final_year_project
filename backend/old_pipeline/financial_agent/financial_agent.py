import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from backend.old_pipeline.financial_agent.ratio_engine import calculate_ratios


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "financial_db", "financial_records.csv")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class FinancialAgent:
    def __init__(self, db_path=DB_PATH):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                "financial_records.csv not found. Run build_financial_database.py first."
            )

        self.df = pd.read_csv(db_path)

    def get_company_year_data(self, company, year, report_type=None):
        company = company.upper()

        data = self.df[
            (self.df["company"] == company) &
            (self.df["year"] == int(year))
        ]

        if report_type:
            data = data[data["report_type"] == report_type]

        values = {}

        for _, row in data.iterrows():
            item = row["item"]
            value = row["value"]

            if item not in values:
                values[item] = value

        return values

    def get_trend(self, company, item):
        company = company.upper()

        data = self.df[
            (self.df["company"] == company) &
            (self.df["item"].str.lower() == item.lower())
        ]

        if data.empty:
            return {
                "company": company,
                "metric": item,
                "data": []
            }

        grouped = (
            data.groupby("year")["value"]
            .mean()
            .reset_index()
            .sort_values("year")
        )

        return {
            "company": company,
            "metric": item,
            "data": grouped.to_dict(orient="records")
        }

    def answer_question(self, question, company, year=None):
        company = company.upper()

        if year:
            values = self.get_company_year_data(company, year)
            ratios = calculate_ratios(values)
        else:
            values = {}
            ratios = {}

        prompt = f"""
You are a professional financial analyst.

User question:
{question}

Company:
{company}

Year:
{year}

Retrieved exact financial values:
{values}

Calculated ratios:
{ratios}

Instructions:
- Use only the provided values and ratios.
- If data is missing, clearly say it is not available.
- Explain performance professionally.
- Discuss profitability, liquidity, solvency, and cash flow when relevant.
- Do not hallucinate.
- Keep answer clear and concise.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional financial statement analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=350
        )

        return {
            "company": company,
            "year": year,
            "values": values,
            "ratios": ratios,
            "answer": response.choices[0].message.content
        }