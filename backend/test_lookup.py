from financial_agent.financial_agent import FinancialAgent

agent = FinancialAgent()

result = agent.get_financial_value(
    company="BPCL",
    item="Earnings per Share before net taxation",
    year=2023,
    report_year=2023,
    report_type="Interim Report"
)

print(result)