from financial_agent.vector_store import FinancialVectorStore

store = FinancialVectorStore()

print("Searching...")

results = store.search(
    question="What is the EPS of BPCL in interim 2023?",
    company="BPCL",
    report_type="Interim Report",
    report_year=2023,
    year=2023,
    n_results=5
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print("\nDOC:", doc)
    print("META:", meta)