"""
Professional Financial Database Builder (SQLite)
------------------------------------------------
Reads ALL Excel/CSV financial statement outputs under:
    backend/data/outputs/<COMPANY>/<Audited Report | Interim Report>/*.(xlsx|xls|csv)

Creates:
    backend/data/financial_db/financial_statements.db
    backend/data/financial_db/financial_records.csv

Important features:
- Detects all companies automatically from folder names.
- Detects Audited Report and Interim Report automatically from folder/file names.
- Extracts every numeric value from every row, not only revenue or selected items.
- Keeps raw_item so no original line item is lost.
- Adds canonical_item for chatbot/ratio matching.
- Adds statement_type: Balance Sheet, Income Statement, Cash Flow, or Unknown.
- Creates normalized SQLite tables and a flat CSV copy for easy testing.
"""

import os
import re
import math
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# If file is placed in backend/scripts or backend/financial_agent, move back to backend
if BASE_DIR.name.lower() in {"scripts", "financial_agent"}:
    BASE_DIR = BASE_DIR.parent

DATA_OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
FINANCIAL_DB_DIR = BASE_DIR / "data" / "financial_db"
SQLITE_DB = FINANCIAL_DB_DIR / "financial_statements.db"
OUTPUT_CSV = FINANCIAL_DB_DIR / "financial_records.csv"
SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".csv")

# Keep only sensible financial years for your project data.
# This prevents false years such as 1903, 2031, 2073 from note numbers or account codes.
MIN_YEAR = 2015
MAX_YEAR = 2026


# -------------------------------------------------------------------
# CANONICAL ITEM MAPPING
# This is only for standardization. Raw items are still stored fully.
# -------------------------------------------------------------------
CANONICAL_RULES = {
    "Revenue": [
        "revenue", "sales", "gross revenue", "total revenue", "operating revenue",
        "income from sales", "sale of goods", "sales revenue"
    ],
    "Cost of Sales": [
        "cost of sales", "cost of goods sold", "cost of goods", "direct cost"
    ],
    "Gross Profit": ["gross profit"],
    "Other Income": ["other income", "other incomes", "miscellaneous income"],
    "Administrative Expenses": ["administrative expenses", "admin expenses"],
    "Selling and Distribution Expenses": [
        "selling distribution expenses", "selling and distribution expenses",
        "selling & distribution expenses", "distribution expenses"
    ],
    "Finance Income": ["finance income", "interest received", "interest income"],
    "Finance Cost": ["finance cost", "interest paid", "interest expense", "borrowing cost"],
    "Profit Before Tax": ["profit before tax", "profit before taxation", "pbt"],
    "Tax Expenses": ["tax expenses", "income tax", "corporate tax", "tax expense"],
    "Net Profit": [
        "profit for the year", "profit after tax", "net profit", "net income",
        "profit from continuing operations", "profit attributable", "total profit"
    ],
    "Total Comprehensive Income": ["total comprehensive income"],

    "Property Plant and Equipment": [
        "property plant equipment", "property plant and equipment", "ppe",
        "fixed assets", "property plants equipments"
    ],
    "Non Current Assets": ["non-current assets", "non current assets", "total non current assets"],
    "Inventories": ["inventories", "inventory", "stock in trade"],
    "Trade and Other Receivables": [
        "trade other receivables", "trade and other receivables", "trade & other receivables",
        "accounts receivable", "receivables"
    ],
    "Cash and Cash Equivalents": [
        "cash cash equivalents", "cash and cash equivalents", "cash & cash equivalents",
        "cash at bank", "cash in hand", "cash balance"
    ],
    "Current Assets": ["current assets", "total current assets"],
    "Total Assets": ["total assets", "assets total"],

    "Share Capital": ["share capital", "paid up capital", "issued capital"],
    "Reserves": ["reserves", "general reserve", "statutory reserve"],
    "Retained Earnings": ["retained earnings", "accumulated profit", "retained profit"],
    "Total Equity": [
        "total equity", "total equities", "shareholders equity", "shareholder equity",
        "owners equity", "equity attributable"
    ],
    "Trade and Other Payables": [
        "trade other payables", "trade and other payables", "trade & other payables",
        "accounts payable", "payables"
    ],
    "Current Liabilities": ["current liabilities", "total current liabilities"],
    "Non Current Liabilities": [
        "non-current liabilities", "non current liabilities", "total non current liabilities"
    ],
    # IMPORTANT: Keep Total Liabilities separate from Total Equity and Liabilities.
    # These are financially different and must NOT be merged.
    "Total Equity and Liabilities": [
        "total equity and liabilities",
        "total liabilities and equity",
        "equity and liabilities",
        "liabilities and equity"
    ],
    "Total Liabilities": [
        "total liabilities",
        "liabilities total"
    ],

    "Net Cash from Operating Activities": [
        "net cash from operating activities", "net cash generated from operating activities",
        "cash flow from operating activities", "operating cash flow",
        "net cash used in operating activities"
    ],
    "Net Cash from Investing Activities": [
        "net cash from investing activities", "net cash used in investing activities",
        "cash flow from investing activities"
    ],
    "Net Cash from Financing Activities": [
        "net cash from financing activities", "net cash used in financing activities",
        "cash flow from financing activities"
    ],
    "Net Increase in Cash": [
        "net increase in cash", "net decrease in cash", "net increase decrease in cash"
    ],
    "Opening Cash Balance": [
        "opening balance of cash", "cash at beginning", "cash and cash equivalents at beginning"
    ],
    "Closing Cash Balance": [
        "closing balance of cash", "cash at end", "cash and cash equivalents at end"
    ],
}


# -------------------------------------------------------------------
# TEXT / NUMBER HELPERS
# -------------------------------------------------------------------
def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slug_text(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_item(text: str) -> str:
    text = clean_text(text)
    # remove leading note numbers only, but do not destroy item names
    text = re.sub(r"^[\d\.\-\)\(\s]+", "", text)
    text = text.replace("&", "and")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_item(item: str) -> str:
    """
    Convert raw financial item names into standard names.

    This function is intentionally strict for TOTAL lines because:
    - Total Liabilities is NOT the same as Total Equity and Liabilities.
    - Total Assets is NOT the same as Current Assets.
    - Current Liabilities is NOT the same as Total Liabilities.

    Raw items are still stored separately, so no original row is lost.
    """
    item_norm = slug_text(item)
    if not item_norm:
        return "Unknown Item"

    # 1) Exact match first. This prevents dangerous partial matches.
    for canonical, patterns in CANONICAL_RULES.items():
        for pattern in patterns:
            if item_norm == slug_text(pattern):
                return canonical

    # 2) Very strict priority rules for important totals.
    # Order matters here. Longer/more specific terms must come first.
    if item_norm in {
        "total equity and liabilities",
        "total liabilities and equity",
        "equity and liabilities",
        "liabilities and equity",
    }:
        return "Total Equity and Liabilities"

    if item_norm in {"total liabilities", "liabilities total"}:
        return "Total Liabilities"

    if item_norm in {"total equity", "total equities", "shareholders equity", "shareholder equity"}:
        return "Total Equity"

    if item_norm in {"total assets", "assets total"}:
        return "Total Assets"

    if item_norm in {"current liabilities", "total current liabilities"}:
        return "Current Liabilities"

    if item_norm in {"current assets", "total current assets"}:
        return "Current Assets"

    # 3) Controlled fuzzy match for non-dangerous items only.
    # Avoid partial matching for totals to prevent wrong financial ratios.
    dangerous_words = {"total", "liabilities", "equity", "assets", "current", "non current"}

    for canonical, patterns in CANONICAL_RULES.items():
        canonical_slug = slug_text(canonical)
        if any(word in canonical_slug for word in dangerous_words):
            continue

        for pattern in patterns:
            p = slug_text(pattern)
            if p and p in item_norm:
                return canonical

    # If not mapped, keep cleaned original item. This ensures EVERY row is still retained.
    return normalize_item(item)


def parse_number(value) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)

    text = clean_text(value)
    if text == "" or text.lower() in {"nan", "none", "nil", "n/a", "na", "-", "—"}:
        return None

    negative = False

    # Accounting negative format: (123,456)
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace(",", "")
    text = re.sub(r"(?i)nu\.?", "", text)
    text = text.replace("$", "").replace("%", "").strip()

    # Keep only simple numeric strings
    if not re.fullmatch(r"[-+]?\d*\.?\d+", text):
        return None

    number = float(text)
    return -number if negative else number


def extract_year(text) -> Optional[int]:
    text = clean_text(text)
    match = re.search(r"(20\d{2}|19\d{2})", text)
    if not match:
        return None
    year = int(match.group(1))
    if MIN_YEAR <= year <= MAX_YEAR:
        return year
    return None


def infer_company_from_path(path: Path) -> str:
    parts = list(path.parts)
    lowered = [p.lower() for p in parts]

    if "outputs" in lowered:
        idx = lowered.index("outputs")
        if idx + 1 < len(parts):
            return clean_text(parts[idx + 1]).upper()

    # fallback: parent-parent is usually company folder
    return clean_text(path.parent.parent.name).upper()


def infer_report_type(path: Path) -> str:
    p = str(path).lower()
    filename = path.name.lower()

    if (
        "interim" in p
        or "interim report" in p
        or filename.startswith("ir_")
        or filename.startswith("ir-")
        or "_ir_" in filename
    ):
        return "Interim Report"

    if (
        "audited" in p
        or "audited report" in p
        or "annual" in p
        or filename.startswith("aa_")
        or filename.startswith("aa-")
        or "_aa_" in filename
    ):
        return "Audited Report"

    return "Unknown"


# -------------------------------------------------------------------
# STATEMENT TYPE DETECTION
# -------------------------------------------------------------------
def infer_statement_type(sheet_name: str, df: pd.DataFrame) -> str:
    """Detect statement type using sheet name and first rows of the sheet."""
    sample_text = " ".join(map(str, df.head(10).fillna("").values.flatten()))
    combined = slug_text(f"{sheet_name} {sample_text}")

    cash_keywords = [
        "cash flow", "cash flows", "statement of cash flow", "statement of cash flows",
        "operating activities", "investing activities", "financing activities"
    ]
    balance_keywords = [
        "balance sheet", "statement of financial position", "financial position",
        "assets", "liabilities", "equity"
    ]
    income_keywords = [
        "income statement", "statement of comprehensive income", "comprehensive income",
        "profit or loss", "profit and loss", "statement of income", "revenue", "expenses"
    ]

    cash_score = sum(1 for k in cash_keywords if slug_text(k) in combined)
    balance_score = sum(1 for k in balance_keywords if slug_text(k) in combined)
    income_score = sum(1 for k in income_keywords if slug_text(k) in combined)

    scores = {
        "Cash Flow": cash_score,
        "Balance Sheet": balance_score,
        "Income Statement": income_score,
    }

    best_statement, best_score = max(scores.items(), key=lambda x: x[1])
    return best_statement if best_score > 0 else "Unknown"


def refine_statement_type(statement_type: str, raw_item: str, sheet_name: str) -> str:
    """Fallback statement detection using the line item itself."""
    text = slug_text(f"{statement_type} {sheet_name} {raw_item}")
    operational_keywords = [
        "production",
        "despatch",
        "budget vs actual",
        "earnings per share",
        "trek division",
        "cafeteria",
        "restaurant",
        "power cost",
        "cost pmt",
        "segment",
        "division",
        "reserve and surplus",
        "operational",
        "kpi",
        "performance",
    ]

    if any(k in text for k in operational_keywords):
        return "Operational Schedule"
    
    if any(k in text for k in [
        "net cash", "operating activities", "investing activities", "financing activities",
        "cash flow", "opening cash", "closing cash"
    ]):
        return "Cash Flow"

    if any(k in text for k in [
        "total assets", "current assets", "non current assets", "liabilities",
        "equity", "share capital", "retained earnings", "payables", "receivables"
    ]):
        return "Balance Sheet"

    if any(k in text for k in [
        "revenue", "sales", "gross profit", "profit before tax", "profit after tax",
        "net profit", "expenses", "expenditure", "income tax", "corporate tax",
        "cost of sales", "finance cost", "operating profit", "earning per share",
        "eps", "return on investment", "cit", "total income", "total expenditure",
        "financial highlights", "interest"
    ]):
        return "Income Statement"

    return statement_type or "Unknown"


# -------------------------------------------------------------------
# EXCEL READING / EXTRACTION
# -------------------------------------------------------------------
def read_workbook(file_path: Path) -> Dict[str, pd.DataFrame]:
    ext = file_path.suffix.lower()

    if ext == ".csv":
        return {"CSV": pd.read_csv(file_path, header=None, dtype=object)}

    return pd.read_excel(file_path, sheet_name=None, header=None, dtype=object)


def find_header_rows(df: pd.DataFrame, max_rows: int = 10) -> List[List[str]]:
    rows = []

    for r in range(min(max_rows, len(df))):
        row = [clean_text(x) for x in df.iloc[r].tolist()]
        row_text = " ".join(row).lower()
        has_year = any(extract_year(c) for c in row)
        has_header_word = any(w in row_text for w in ["particular", "year", "notes", "note", "amount"])

        if has_year or has_header_word:
            rows.append(row)

    if not rows and len(df) > 0:
        rows.append([clean_text(x) for x in df.iloc[0].tolist()])

    return rows


def find_item_from_row(row_values: List, value_col_idx: int) -> str:
    """Pick nearest meaningful text cell to the left of a numeric value."""

    # Prefer nearest text cell on the left
    for idx in range(value_col_idx - 1, -1, -1):
        text = clean_text(row_values[idx])
        if not text:
            continue
        if parse_number(text) is None and not extract_year(text):
            item = normalize_item(text)
            if item and item.lower() not in {"particulars", "particular", "notes", "note", "amount"}:
                return item

    # Fallback: first non-numeric text in row
    for cell in row_values:
        text = clean_text(cell)
        if not text:
            continue
        if parse_number(text) is None and not extract_year(text):
            item = normalize_item(text)
            if item and item.lower() not in {"particulars", "particular", "notes", "note", "amount"}:
                return item

    return ""


def get_column_name(header_rows: List[List[str]], col_idx: int) -> str:
    """Find best column label/year above the numeric column."""
    for header in reversed(header_rows):
        if col_idx < len(header) and header[col_idx]:
            return header[col_idx]
    return ""


def extract_records_from_file(file_path: Path) -> List[dict]:
    company = infer_company_from_path(file_path)
    report_type = infer_report_type(file_path)
    file_year = extract_year(file_path.name)
    records: List[dict] = []

    try:
        sheets = read_workbook(file_path)
    except Exception as e:
        print(f"[SKIPPED] Could not read {file_path}: {e}")
        return records

    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue

        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue

        base_statement_type = infer_statement_type(sheet_name, df)
        header_rows = find_header_rows(df)

        for row_idx, row in df.iterrows():
            row_values = row.tolist()

            for col_idx, cell in enumerate(row_values):
                value = parse_number(cell)
                if value is None:
                    continue

                raw_item = find_item_from_row(row_values, col_idx)
                if not raw_item:
                    continue

                column_name = get_column_name(header_rows, col_idx)
                year = extract_year(column_name) or file_year

                # Skip values without a valid project financial year.
                # This avoids account codes/note numbers becoming fake years.
                if year is None:
                    continue

                statement_type = refine_statement_type(base_statement_type, raw_item, sheet_name)

                records.append({
                    "company": company,
                    "report_type": report_type,
                    "report_year": file_year or year,
                    "fiscal_year": int(year),
                    "source_file": file_path.name,
                    "statement_type": statement_type,
                    "sheet_name": clean_text(sheet_name),
                    "row_number": int(row_idx) + 1,
                    "raw_item": raw_item,
                    "canonical_item": canonical_item(raw_item),
                    "column_name": column_name,
                    "value": float(value),
                })

    return records


# -------------------------------------------------------------------
# SQLITE DATABASE
# -------------------------------------------------------------------
def connect_db() -> sqlite3.Connection:
    FINANCIAL_DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    DROP TABLE IF EXISTS ratio_results;
    DROP TABLE IF EXISTS financial_values;
    DROP TABLE IF EXISTS line_items;
    DROP TABLE IF EXISTS statements;
    DROP TABLE IF EXISTS reports;
    DROP TABLE IF EXISTS companies;

    CREATE TABLE companies (
        company_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL UNIQUE,
        company_name TEXT
    );

    CREATE TABLE reports (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL,
        report_type TEXT NOT NULL,
        report_year INTEGER,
        source_file TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(company_id, report_type, report_year, source_file),
        FOREIGN KEY(company_id) REFERENCES companies(company_id)
    );

    CREATE TABLE statements (
        statement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_id INTEGER NOT NULL,
        statement_type TEXT NOT NULL,
        sheet_name TEXT NOT NULL,
        UNIQUE(report_id, statement_type, sheet_name),
        FOREIGN KEY(report_id) REFERENCES reports(report_id)
    );

    CREATE TABLE line_items (
        line_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_item TEXT NOT NULL,
        canonical_item TEXT NOT NULL,
        item_key TEXT NOT NULL UNIQUE
    );

    CREATE TABLE financial_values (
        value_id INTEGER PRIMARY KEY AUTOINCREMENT,
        statement_id INTEGER NOT NULL,
        line_item_id INTEGER NOT NULL,
        fiscal_year INTEGER NOT NULL,
        column_name TEXT,
        row_number INTEGER,
        value REAL NOT NULL,
        UNIQUE(statement_id, line_item_id, fiscal_year, column_name, row_number, value),
        FOREIGN KEY(statement_id) REFERENCES statements(statement_id),
        FOREIGN KEY(line_item_id) REFERENCES line_items(line_item_id)
    );

    CREATE TABLE ratio_results (
        ratio_id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL,
        report_type TEXT,
        fiscal_year INTEGER NOT NULL,
        ratio_name TEXT NOT NULL,
        ratio_value REAL,
        calculated_at TEXT NOT NULL,
        UNIQUE(company_id, report_type, fiscal_year, ratio_name),
        FOREIGN KEY(company_id) REFERENCES companies(company_id)
    );

    CREATE INDEX idx_values_year ON financial_values(fiscal_year);
    CREATE INDEX idx_items_canonical ON line_items(canonical_item);
    CREATE INDEX idx_reports_type_year ON reports(report_type, report_year);
    CREATE INDEX idx_statements_type ON statements(statement_type);
    """)
    conn.commit()


def get_or_create(
    conn: sqlite3.Connection,
    select_sql: str,
    insert_sql: str,
    select_params: tuple,
    insert_params: tuple,
) -> int:
    row = conn.execute(select_sql, select_params).fetchone()
    if row:
        return int(row[0])
    cur = conn.execute(insert_sql, insert_params)
    return int(cur.lastrowid)


def insert_records(conn: sqlite3.Connection, records: List[dict]) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")

    for r in records:
        company_id = get_or_create(
            conn,
            "SELECT company_id FROM companies WHERE ticker=?",
            "INSERT INTO companies(ticker, company_name) VALUES(?, ?)",
            (r["company"],),
            (r["company"], r["company"]),
        )

        report_id = get_or_create(
            conn,
            """
            SELECT report_id FROM reports
            WHERE company_id=? AND report_type=? AND report_year IS ? AND source_file=?
            """,
            """
            INSERT INTO reports(company_id, report_type, report_year, source_file, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (company_id, r["report_type"], r["report_year"], r["source_file"]),
            (company_id, r["report_type"], r["report_year"], r["source_file"], now),
        )

        statement_id = get_or_create(
            conn,
            "SELECT statement_id FROM statements WHERE report_id=? AND statement_type=? AND sheet_name=?",
            "INSERT INTO statements(report_id, statement_type, sheet_name) VALUES(?, ?, ?)",
            (report_id, r["statement_type"], r["sheet_name"]),
            (report_id, r["statement_type"], r["sheet_name"]),
        )

        # item_key includes canonical + raw to avoid losing similar but distinct items.
        # This keeps all rows while still supporting canonical matching.
        item_key = slug_text(f'{r["canonical_item"]} {r["raw_item"]}')
        line_item_id = get_or_create(
            conn,
            "SELECT line_item_id FROM line_items WHERE item_key=?",
            "INSERT INTO line_items(raw_item, canonical_item, item_key) VALUES(?, ?, ?)",
            (item_key,),
            (r["raw_item"], r["canonical_item"], item_key),
        )

        conn.execute(
            """
            INSERT OR IGNORE INTO financial_values
            (statement_id, line_item_id, fiscal_year, column_name, row_number, value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                statement_id,
                line_item_id,
                r["fiscal_year"],
                r["column_name"],
                r["row_number"],
                r["value"],
            ),
        )

    conn.commit()


def export_flat_csv(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT
        c.ticker AS company,
        r.report_type,
        r.report_year,
        fv.fiscal_year AS year,
        r.source_file,
        s.statement_type,
        s.sheet_name,
        fv.row_number,
        li.raw_item,
        li.canonical_item AS item,
        fv.column_name,
        fv.value
    FROM financial_values fv
    JOIN line_items li ON li.line_item_id = fv.line_item_id
    JOIN statements s ON s.statement_id = fv.statement_id
    JOIN reports r ON r.report_id = s.report_id
    JOIN companies c ON c.company_id = r.company_id
    ORDER BY
        c.ticker,
        r.report_type,
        fv.fiscal_year,
        r.source_file,
        s.statement_type,
        fv.row_number;
    """
    df = pd.read_sql_query(query, conn)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    return df


def print_quality_summary(df: pd.DataFrame) -> None:
    print("\nQuality Summary")
    print("---------------")
    print(f"Total extracted values: {len(df)}")

    if df.empty:
        return

    print(f"Companies: {', '.join(sorted(df['company'].dropna().unique()))}")

    print("\nReport types:")
    print(df["report_type"].value_counts(dropna=False).to_string())

    print("\nStatement types:")
    print(df["statement_type"].value_counts(dropna=False).to_string())

    unknown_statement_count = int((df["statement_type"] == "Unknown").sum())
    if unknown_statement_count > 0:
        print(f"\nWarning: {unknown_statement_count} values have Unknown statement_type.")
        print("This is not fatal. It means the sheet/item did not clearly say Balance Sheet, Income Statement, or Cash Flow.")


# -------------------------------------------------------------------
# MAIN BUILD FUNCTION
# -------------------------------------------------------------------
def build_financial_database(data_outputs_dir: Path = DATA_OUTPUTS_DIR) -> pd.DataFrame:
    if not data_outputs_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_outputs_dir}")

    files = [
        p for p in data_outputs_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$")
    ]

    if not files:
        raise FileNotFoundError(f"No Excel/CSV files found under: {data_outputs_dir}")

    all_records: List[dict] = []

    for file_path in files:
        print(f"[PROCESSING] {file_path}")
        records = extract_records_from_file(file_path)
        all_records.extend(records)

    conn = connect_db()
    create_schema(conn)
    insert_records(conn, all_records)
    df = export_flat_csv(conn)
    conn.close()

    print("\nFinancial SQLite database created successfully.")
    print(f"Database: {SQLITE_DB}")
    print(f"CSV copy: {OUTPUT_CSV}")
    print_quality_summary(df)

    return df


if __name__ == "__main__":
    build_financial_database()
