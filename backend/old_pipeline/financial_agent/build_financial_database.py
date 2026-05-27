"""
Full Financial Database Builder
--------------------------------
Reads ALL rows from ALL Excel sheets under:
    backend/data/outputs/<COMPANY>/<Audited Report | Interim Report>/*.xlsx

Creates:
    backend/data/financial_db/financial_records.csv

Each numeric cell becomes one searchable record:
company, report_type, year, source_file, sheet_name, row_number,
raw_item, item, column_name, value
"""

import os
import re
import math
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUTPUTS_DIR = os.path.join(BASE_DIR, "data", "outputs")
FINANCIAL_DB_DIR = os.path.join(BASE_DIR, "data", "financial_db")
OUTPUT_CSV = os.path.join(FINANCIAL_DB_DIR, "financial_records.csv")

SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".csv")


def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.strip()


def normalize_item(text):
    text = clean_text(text)
    text = re.sub(r"^[\d\.\-\)\(\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_number(value):
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none", "nil", "n/a", "na", "-"]:
        return None

    # Detect bracketed negatives: (123,456)
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace(",", "")
    text = text.replace("Nu.", "").replace("Nu", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = text.strip()

    # Keep only simple numeric strings
    if not re.fullmatch(r"[-+]?\d*\.?\d+", text):
        return None

    try:
        number = float(text)
        return -number if negative else number
    except Exception:
        return None


def extract_year_from_text(text):
    if text is None:
        return None
    match = re.search(r"(20\d{2}|19\d{2})", str(text))
    if match:
        return int(match.group(1))
    return None


def infer_year_from_file(path):
    return extract_year_from_text(os.path.basename(path))


def infer_report_type(path):
    p = path.lower()
    if "interim" in p:
        return "Interim Report"
    if "audited" in p or "annual" in p:
        return "Audited Report"
    return "Unknown"


def infer_company_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    try:
        idx = parts.index("outputs")
        return parts[idx + 1].upper().strip()
    except Exception:
        # fallback: parent-parent folder name
        return os.path.basename(os.path.dirname(os.path.dirname(path))).upper().strip()


def find_item_from_row(row_values, value_col_idx):
    """Pick the nearest meaningful text cell to the left of the numeric value."""
    # Prefer nearest text cell on the left
    for idx in range(value_col_idx - 1, -1, -1):
        text = clean_text(row_values[idx])
        if text and parse_number(text) is None:
            return normalize_item(text)

    # Fallback: first non-numeric text in row
    for cell in row_values:
        text = clean_text(cell)
        if text and parse_number(text) is None:
            return normalize_item(text)

    return ""


def read_workbook(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return {"CSV": pd.read_csv(file_path, header=None, dtype=object)}

    return pd.read_excel(
        file_path,
        sheet_name=None,
        header=None,
        dtype=object,
        engine="openpyxl" if ext == ".xlsx" else None
    )


def extract_records_from_file(file_path):
    records = []
    company = infer_company_from_path(file_path)
    report_type = infer_report_type(file_path)
    file_year = infer_year_from_file(file_path)
    source_file = os.path.basename(file_path)

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

        # Header hints from first few rows are useful when column names are years
        header_candidates = []
        for r in range(min(5, len(df))):
            header_candidates.append([clean_text(x) for x in df.iloc[r].tolist()])

        for row_idx, row in df.iterrows():
            row_values = row.tolist()

            for col_idx, cell in enumerate(row_values):
                numeric_value = parse_number(cell)
                if numeric_value is None:
                    continue

                raw_item = find_item_from_row(row_values, col_idx)
                if not raw_item:
                    continue

                column_name = ""
                # Find best header above current numeric column
                for header_row in reversed(header_candidates):
                    if col_idx < len(header_row) and header_row[col_idx]:
                        column_name = header_row[col_idx]
                        break

                year = extract_year_from_text(column_name) or file_year

                # Skip values with no year because chatbot mostly queries by year
                if year is None:
                    continue

                records.append({
                    "company": company,
                    "report_type": report_type,
                    "year": int(year),
                    "source_file": source_file,
                    "sheet_name": str(sheet_name),
                    "row_number": int(row_idx) + 1,
                    "raw_item": raw_item,
                    "item": normalize_item(raw_item),
                    "column_name": column_name,
                    "value": numeric_value,
                })

    return records


def build_financial_database():
    if not os.path.exists(DATA_OUTPUTS_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_OUTPUTS_DIR}")

    all_records = []

    for root, _, files in os.walk(DATA_OUTPUTS_DIR):
        for filename in files:
            if filename.startswith("~$"):
                continue
            if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            file_path = os.path.join(root, filename)
            print(f"[PROCESSING] {file_path}")
            records = extract_records_from_file(file_path)
            all_records.extend(records)

    os.makedirs(FINANCIAL_DB_DIR, exist_ok=True)

    df = pd.DataFrame(all_records)

    if not df.empty:
        df = df.drop_duplicates(
            subset=[
                "company", "report_type", "year", "source_file",
                "sheet_name", "row_number", "raw_item", "column_name", "value"
            ]
        )
        df = df.sort_values(["company", "report_type", "year", "source_file", "sheet_name", "row_number"])

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nFinancial database created successfully.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total records: {len(df)}")

    if not df.empty:
        print("Companies:", ", ".join(sorted(df["company"].unique())))

    return df


if __name__ == "__main__":
    build_financial_database()
