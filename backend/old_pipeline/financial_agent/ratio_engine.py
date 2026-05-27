def safe_divide(a, b):
    if a is None or b is None or b == 0:
        return None

    return round(a / b, 4)


def calculate_ratios(values):
    revenue = values.get("Revenue")
    net_profit = values.get("Net Profit")
    total_assets = values.get("Total Assets")
    total_liabilities = values.get("Total Liabilities")
    total_equity = values.get("Total Equity")
    current_assets = values.get("Current Assets")
    current_liabilities = values.get("Current Liabilities")
    operating_cash_flow = values.get("Operating Cash Flow")

    if total_equity is None and total_assets and total_liabilities:
        total_equity = total_assets - total_liabilities

    ratios = {
        "Net Profit Margin": safe_divide(net_profit, revenue),
        "Return on Assets": safe_divide(net_profit, total_assets),
        "Return on Equity": safe_divide(net_profit, total_equity),
        "Current Ratio": safe_divide(current_assets, current_liabilities),
        "Debt Ratio": safe_divide(total_liabilities, total_assets),
        "Debt to Equity": safe_divide(total_liabilities, total_equity),
        "Operating Cash Ratio": safe_divide(operating_cash_flow, current_liabilities)
    }

    return ratios