import pandas as pd

class TrendEngine:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def get_item_trend(self, company, item, statement_type=None, report_type=None):
        company = company.upper()

        data = self.df[
            (self.df["company"] == company) &
            (self.df["item"].str.lower() == item.lower())
        ]

        if statement_type:
            data = data[data["statement_type"] == statement_type]

        if report_type:
            data = data[data["report_type"] == report_type]

        data = data[
            (data["year"] >= 2015) &
            (data["year"] <= 2026)
        ]

        if data.empty:
            return []

        grouped = (
            data.groupby("year")["value"]
            .mean()
            .reset_index()
            .sort_values("year")
        )

        trend = []
        previous = None

        for _, row in grouped.iterrows():
            year = int(row["year"])
            value = float(row["value"])

            growth = None
            if previous is not None and previous != 0:
                growth = round(((value - previous) / abs(previous)) * 100, 2)

            trend.append({
                "year": year,
                "value": round(value, 2),
                "growth_percent": growth
            })

            previous = value

        return trend

    def summarize_trend(self, trend_data, metric_name):
        if not trend_data:
            return f"No trend data available for {metric_name}."

        first = trend_data[0]["value"]
        last = trend_data[-1]["value"]

        overall_growth = None
        if first != 0:
            overall_growth = round(((last - first) / abs(first)) * 100, 2)

        years = len(trend_data)

        if overall_growth is None:
            return f"{metric_name} trend is available for {years} years."

        direction = "increased" if overall_growth >= 0 else "decreased"

        return (
            f"{metric_name} has {direction} by "
            f"{abs(overall_growth)}% over {years} years."
        )