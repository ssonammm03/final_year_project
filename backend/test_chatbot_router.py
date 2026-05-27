from financial_agent.financial_agent import FinancialAgent

agent = FinancialAgent()

questions = [
    "What is the EPS of BPCL in interim 2023?",
    "What is the audited revenue of BBPL in 2024?",
    "Show me BBPL revenue trend",
    "Give me BBPL financial performance summary",
]

for q in questions:
    print("\nQUESTION:", q)
    result = agent.answer_user_question(q)
    print("ANSWER:", result["answer"])