from financial_agent.financial_agent import FinancialAgent

agent = FinancialAgent()

result = agent.answer_question(
    question="Give me a financial performance summary",
    company="BBPL",
    year=2015
)

print(result["answer"])
print("\nRatios:")
print(result["ratios"])