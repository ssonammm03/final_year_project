from pathlib import Path
import pandas as pd
import chromadb


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "financial_db" / "financial_records.csv"
CHROMA_DIR = BASE_DIR / "data" / "vector_db"


class FinancialVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="financial_records"
        )

    def build_index(self, limit=None):
        df = pd.read_csv(CSV_PATH)

        if limit:
            df = df.head(limit)

        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            doc = (
                f"Company: {row['company']}. "
                f"Report type: {row['report_type']}. "
                f"Report year: {row['report_year']}. "
                f"Fiscal year: {row['year']}. "
                f"Statement type: {row['statement_type']}. "
                f"Sheet: {row['sheet_name']}. "
                f"Financial item: {row['item']}. "
                f"Raw item: {row['raw_item']}. "
                f"Column: {row['column_name']}. "
                f"Value: {row['value']}."
            )

            documents.append(doc)

            metadatas.append({
                "company": str(row["company"]),
                "report_type": str(row["report_type"]),
                "report_year": int(row["report_year"]) if not pd.isna(row["report_year"]) else 0,
                "year": int(row["year"]),
                "statement_type": str(row["statement_type"]),
                "item": str(row["item"]),
                "source_file": str(row["source_file"]),
            })

            ids.append(f"financial_row_{idx}")

        # Chroma can be slow if adding too many at once
        batch_size = 200

        for start in range(0, len(documents), batch_size):
            end = start + batch_size

            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        return {
            "status": "success",
            "indexed_records": len(documents),
            "vector_db_path": str(CHROMA_DIR)
        }

    def search(
        self,
        question,
        company=None,
        report_type=None,
        report_year=None,
        year=None,
        n_results=5,
    ):

        filters = []

        if company:
            filters.append({"company": company.upper()})

        if report_type:
            filters.append({"report_type": report_type})

        if report_year:
            filters.append({"report_year": int(report_year)})

        if year:
            filters.append({"year": int(year)})

        if len(filters) == 0:
            where = None
        elif len(filters) == 1:
            where = filters[0]
        else:
            where = {"$and": filters}

        results = self.collection.query(
            query_texts=[question],
            n_results=n_results * 3,
            where=where,
        )

        cleaned_docs = []
        cleaned_meta = []

        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):

            doc_lower = doc.lower()

            if any(x in doc_lower for x in [
                "column: sl. no.",
                "column: sl no",
                "column: s.no",
                "column: no.",
            ]):
                continue

            cleaned_docs.append(doc)
            cleaned_meta.append(meta)

            if len(cleaned_docs) >= n_results:
                break

        return {
            "documents": [cleaned_docs],
            "metadatas": [cleaned_meta]
        }