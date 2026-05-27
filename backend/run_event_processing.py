import os
import glob

from features.event_feature_engineer import (
    process_company_file
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed_event"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


COMPANIES = [
    "BBPL", "BCCL", "BFAL", "BIL", "BNBL",
    "BPCL", "BTCL", "DFAL", "DPL", "DPNB",
    "DWAL", "GICB", "KCL", "PCAL", "RICB",
    "STCB", "TBL"
]


def get_latest_csv(company):

    company_dir = os.path.join(
        RAW_DIR,
        company
    )

    csv_files = glob.glob(
        os.path.join(company_dir, "*.csv")
    )

    if not csv_files:
        return None

    latest_file = max(
        csv_files,
        key=os.path.getmtime
    )

    return latest_file


def process_all_companies():

    for company in COMPANIES:

        try:

            csv_path = get_latest_csv(company)

            if csv_path is None:

                print(f"No CSV found for {company}")
                continue

            print("\n")
            print("=" * 70)
            print(f"PROCESSING {company}")
            print("=" * 70)

            processed_df = process_company_file(
                csv_path,
                company
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{company}_event_featured.csv"
            )

            processed_df.to_csv(
                output_path,
                index=False
            )

            print(f"Saved: {output_path}")

        except Exception as e:

            print(f"[ERROR] {company}: {e}")


if __name__ == "__main__":

    process_all_companies()