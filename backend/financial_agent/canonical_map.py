import re


CANONICAL_ITEMS = {
    "revenue": [
        "revenue", "sales", "gross revenue", "total revenue",
        "operating revenue", "income from sales"
    ],
    "gross_profit": [
        "gross profit"
    ],
    "net_profit": [
        "net profit", "profit after tax", "profit for the year",
        "net income", "profit attributable to shareholders",
        "total comprehensive income"
    ],
    "total_assets": [
        "total assets", "assets total"
    ],
    "total_liabilities": [
        "total liabilities", "liabilities total"
    ],
    "total_equity": [
        "total equity", "shareholders equity", "owners equity",
        "equity attributable to owners"
    ],
    "current_assets": [
        "current assets", "total current assets"
    ],
    "current_liabilities": [
        "current liabilities", "total current liabilities"
    ],
    "cash_and_cash_equivalents": [
        "cash and cash equivalents", "cash at bank", "cash in hand",
        "cash balance"
    ],
    "operating_cash_flow": [
        "net cash from operating activities",
        "cash flow from operating activities",
        "operating cash flow",
        "net cash generated from operating activities"
    ],
    "opening_cash_balance": [
        "cash and cash equivalents at beginning",
        "opening cash balance",
        "cash at beginning of year"
    ],
    "closing_cash_balance": [
        "cash and cash equivalents at end",
        "closing cash balance",
        "cash at end of year"
    ],
}


def clean_item_name(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_item(item):
    cleaned = clean_item_name(item)

    for canonical, aliases in CANONICAL_ITEMS.items():
        for alias in aliases:
            alias_cleaned = clean_item_name(alias)

            if cleaned == alias_cleaned:
                return canonical

            if alias_cleaned in cleaned:
                return canonical

    return cleaned.replace(" ", "_")