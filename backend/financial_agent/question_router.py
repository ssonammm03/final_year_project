import re
import difflib
from openai import OpenAI
import os

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None


def detect_company(question, companies):
    q = question.upper()

    for company in companies:
        company = str(company).upper().strip()
        if re.search(rf"\b{re.escape(company)}\b", q):
            return company

    return None


def detect_year(question):
    match = re.search(r"\b(20\d{2}|19\d{2}|207\d|208\d)\b", question)
    if match:
        return int(match.group(1))
    return None


def detect_report_type(question):
    q = question.lower()

    if "interim" in q:
        return "Interim Report"

    if "audited" in q or "annual" in q:
        return "Audited Report"

    return None


def detect_intent(question):
    q = question.lower()

    if any(word in q for word in ["trend", "growth", "increase", "decrease", "over time", "performance trend"]):
        return "trend"

    if any(word in q for word in ["summary", "performance", "overview", "financial health", "how is", "analyze", "analyse"]):
        return "summary"

    return "lookup"


def detect_statement_type(item):
    if not item:
        return None

    q = item.lower()

    income_items = [
        "revenue", "sales", "profit", "income", "expense", "cost",
        "eps", "earning", "tax", "dividend"
    ]

    balance_items = [
        "asset", "liability", "liabilities", "equity", "capital",
        "receivable", "payable", "inventory", "cash and cash equivalents"
    ]

    cashflow_items = [
        "cash flow", "operating activities", "investing activities",
        "financing activities", "net cash"
    ]

    if any(word in q for word in cashflow_items):
        return "Cash Flow"

    if any(word in q for word in balance_items):
        return "Balance Sheet"

    if any(word in q for word in income_items):
        return "Income Statement"

    return None


def detect_item(question, valid_items_from_db=None):
    q = question.lower().strip()

    if valid_items_from_db:
        valid_items_from_db = [
            str(item).strip()
            for item in valid_items_from_db
            if item is not None and str(item).strip() and str(item).lower() != "nan"
        ]

        # EPS detection
    if (
        "eps" in q or
        "earning per share" in q or
        "earnings per share" in q
    ):

        if valid_items_from_db:

            eps_priority = [
                "earning per share",
                "earnings per share",
                "basic",
                "diluted",
                "eps"
            ]

             # Smart contains matching
            for db_item in valid_items_from_db:

                db_lower = str(db_item).lower()

                if any(keyword in db_lower for keyword in eps_priority):
                    return db_item

            # Fuzzy fallback
            matches = difflib.get_close_matches(
                "earning per share",
                valid_items_from_db,
                n=1,
                cutoff=0.5
            )

            if matches:
                return matches[0]

        return "EARNING PER SHARE (Basic & Diluted)"
    

    item_map = {
        "cost of sales": "Cost of Sales",
        "cost of slaes": "Cost of Sales",
        "gross profit": "Gross Profit",
        "profit after tax": "Net Profit",
        "net profit": "Net Profit",
        "profit before tax": "Profit Before Tax",
        "total assets": "Total Assets",
        "total asset": "Total Assets",
        "total liabilities": "Total Liabilities",
        "total liability": "Total Liabilities",
        "total libality": "Total Liabilities",
        "libality": "Total Liabilities",
        "total equity": "Total Equity",
        "operating cash flow": "Net Cash from Operating Activities",
        "cash flow from operating": "Net Cash from Operating Activities",
        "net cash from operating": "Net Cash from Operating Activities",
        "revenue": "Revenue",
        "sales": "Revenue",
        "profit": "Net Profit",
        "liability": "Total Liabilities",
        "liabilities": "Total Liabilities",
        "asset": "Total Assets",
        "assets": "Total Assets",
        "equity": "Total Equity",
    }

    for key in sorted(item_map.keys(), key=len, reverse=True):
        if key in q:
            return item_map[key]

    words = q.split()
    for word in words:
        matches = difflib.get_close_matches(word, item_map.keys(), n=1, cutoff=0.7)
        if matches:
            return item_map[matches[0]]

    if openai_client and valid_items_from_db:
        try:
            question_words = set(re.findall(r"\w+", q))

            candidate_pool = [
                item for item in valid_items_from_db
                if any(word in str(item).lower() for word in question_words if len(word) > 3)
            ][:50]

            if not candidate_pool:
                candidate_pool = valid_items_from_db[:40]

            prompt = f"""
You are an expert financial schema mapper. Identify if the user is asking for a specific financial item from the data pool below.

User Question: "{question}"

Available Data Items:
{candidate_pool}

Instructions:
- Return ONLY the exact string of the matching item from the list.
- If there is no reasonable match, return the exact word "NONE".
- Do not add explanations, punctuation, or markdown formatting.
"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=30
            )

            extracted_item = response.choices[0].message.content.strip()

            if extracted_item and extracted_item != "NONE" and extracted_item in valid_items_from_db:
                return extracted_item

        except Exception:
            pass

    return None