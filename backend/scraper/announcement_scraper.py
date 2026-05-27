import os
import re
import time
import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVENT_DIR = os.path.join(BASE_DIR, "data", "events")
os.makedirs(EVENT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(EVENT_DIR, "dividend_events.csv")


COMPANIES = [
    "BBPL", "BCCL", "BFAL", "BIL", "BNBL",
    "BPCL", "BTCL", "DFAL", "DPL", "DPNB",
    "DWAL", "GICB", "KCL", "PCAL", "RICB",
    "STCB", "TBL"
]


def detect_company(text):
    text = text.upper()

    for company in COMPANIES:
        if company in text:
            return company

    return None


def extract_dividend_percent(text):
    match = re.search(r'(\d+(\.\d+)?)\s*%', text)

    if match:
        return float(match.group(1))

    return None


def extract_bonus_ratio(text):
    match = re.search(r'1\s*:\s*\d+', text)

    if match:
        return match.group(0)

    return None


def extract_ex_date(text):
    match = re.search(
        r'Ex-Date\s*(\d{1,2}\s\w+)',
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


def extract_record_date(text):
    match = re.search(
        r'Record Date\s*(\d{1,2}\s\w+)',
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


def is_dividend_related(text):
    keywords = [
        "dividend",
        "bonus share",
        "agm",
        "record date",
        "ex-date"
    ]

    text = text.lower()

    return any(keyword in text for keyword in keywords)


def scrape_announcements():

    url = "https://rsebl.org.bt/news-and-announcements"

    all_records = []

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=False)

        page = browser.new_page()

        page.goto(url, timeout=60000)

        page.wait_for_timeout(5000)

        page_number = 1

        while True:

            print("=" * 60)
            print(f"SCRAPING PAGE {page_number}")
            print("=" * 60)

            page.wait_for_timeout(3000)

            cards = page.locator("div.rounded-xl")

            count = cards.count()

            print(f"Found {count} cards")

            for i in range(count):

                try:

                    card = cards.nth(i)

                    text = card.inner_text()

                    if not is_dividend_related(text):
                        continue

                    company = detect_company(text)

                    if company is None:
                        continue

                    dividend_percent = extract_dividend_percent(text)

                    bonus_ratio = extract_bonus_ratio(text)

                    ex_date = extract_ex_date(text)

                    record_date = extract_record_date(text)

                    date_match = re.search(
                        r'(\d{1,2}\s\w+\s\d{4})',
                        text
                    )

                    announcement_date = None

                    if date_match:
                        try:
                            announcement_date = datetime.strptime(
                                date_match.group(1),
                                "%d %B %Y"
                            ).strftime("%Y-%m-%d")

                        except:
                            pass

                    lines = text.split("\n")

                    title = lines[1] if len(lines) > 1 else ""

                    record = {
                        "company": company,
                        "announcement_date": announcement_date,
                        "title": title,
                        "dividend_percent": dividend_percent,
                        "bonus_ratio": bonus_ratio,
                        "ex_date": ex_date,
                        "record_date": record_date,
                        "raw_text": text
                    }

                    all_records.append(record)

                    print(record)

                except Exception as e:
                    print(f"Error parsing card {i}: {e}")

            # NEXT PAGE
            try:

                next_button = page.locator(
                    "button:has-text('Next')"
                ).first

                if next_button.is_disabled():
                    print("No more pages.")
                    break

                print("Moving to next page...")

                next_button.click()

                time.sleep(4)

                page_number += 1

            except Exception as e:

                print("Pagination finished.")
                print(e)

                break

        browser.close()

    df = pd.DataFrame(all_records)

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort
    df = df.sort_values(
        by=["company", "announcement_date"],
        ascending=[True, False]
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved dividend announcements:")
    print(OUTPUT_FILE)

    print(f"\nTotal records: {len(df)}")

    return df


if __name__ == "__main__":
    scrape_announcements()