from financial_agent.financial_agent import FinancialAgent

agent = FinancialAgent()

questions = [
    "What is the EPS of BPCL in interim 2023?",
    "What is the audited revenue trend of BBPL?",
    "Give me BBPL financial performance summary",
]

for q in questions:
    print("\nQUESTION:", q)
    result = agent.answer_user_question(q)

    print("\nANSWER:")
    print(result.get("answer"))

    print("\nEXPLANATION:")
    print(result.get("explanation", "No explanation generated."))