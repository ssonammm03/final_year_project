from financial_agent.financial_agent import FinancialAgent

agent = FinancialAgent()

result = agent.analyze_company_performance_trends(
    company="BBPL",
    report_type="Audited Report"
)

for metric, data in result["results"].items():
    print("\n" + metric)
    print(data["summary"])
    for row in data["trend"]:
        print(row)