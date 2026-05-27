def safe_divide(a, b):
    if a is None or b is None or b == 0:
        return None
    return round(a / b, 4)


def calculate_ratios(values):
    revenue = values.get("revenue")
    net_profit = values.get("net_profit")
    total_assets = values.get("total_assets")
    total_liabilities = values.get("total_liabilities")
    total_equity = values.get("total_equity")
    current_assets = values.get("current_assets")
    current_liabilities = values.get("current_liabilities")
    operating_cash_flow = values.get("operating_cash_flow")

    if total_equity is None and total_assets is not None and total_liabilities is not None:
        total_equity = total_assets - total_liabilities

    return {
        "Net Profit Margin %": round(safe_divide(net_profit, revenue) * 100, 4) if safe_divide(net_profit, revenue) else None,
        "Return on Assets %": round(safe_divide(net_profit, total_assets) * 100, 4) if safe_divide(net_profit, total_assets) else None,
        "Return on Equity %": round(safe_divide(net_profit, total_equity) * 100, 4) if safe_divide(net_profit, total_equity) else None,
        "Current Ratio": safe_divide(current_assets, current_liabilities),
        "Debt Ratio %": round(safe_divide(total_liabilities, total_assets) * 100, 4) if safe_divide(total_liabilities, total_assets) else None,
        "Debt to Equity": safe_divide(total_liabilities, total_equity),
        "Operating Cash Ratio": safe_divide(operating_cash_flow, current_liabilities),
    }