from financial_agent.financial_agent import FinancialAgent

agent = FinancialAgent()

result = agent.analyze_trend(
    company="BBPL",
    item="Revenue",
    statement_type="Income Statement",
    report_type="Audited Report"
)

print(result["summary"])

for row in result["trend"]:
    print(row)