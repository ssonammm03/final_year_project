import os
import time
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def download_stock_data(company: str, headless: bool = False, retries: int = 3):
    company = company.upper()
    url = f"https://rsebl.org.bt/stocks/{company}"

    company_dir = os.path.join(RAW_DATA_DIR, company)
    os.makedirs(company_dir, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(company_dir, f"{company}_{today}.csv")

    for attempt in range(1, retries + 1):
        try:
            print(f"Opening {company} page... Attempt {attempt}/{retries}")

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()

                page.goto(url, timeout=60000, wait_until="domcontentloaded")
                page.wait_for_load_state("networkidle", timeout=60000)

                print("Looking for Download button...")

                download_selectors = [
                    "button:has-text('Download')",
                    "text=Download",
                    "button:has-text('download')",
                    "[role='button']:has-text('Download')"
                ]

                download_button = None

                for selector in download_selectors:
                    try:
                        locator = page.locator(selector).first
                        locator.wait_for(timeout=5000)
                        download_button = locator
                        break
                    except Exception:
                        continue

                if download_button is None:
                    print(f"[ERROR] Download button not found for {company}")
                    browser.close()
                    continue

                download_button.click()
                time.sleep(1.5)

                print("Looking for All Data option...")

                all_data_selectors = [
                    "text=All Data",
                    "button:has-text('All Data')",
                    "[role='menuitem']:has-text('All Data')",
                    "[role='option']:has-text('All Data')"
                ]

                all_data_button = None

                for selector in all_data_selectors:
                    try:
                        locator = page.locator(selector).first
                        locator.wait_for(timeout=5000)
                        all_data_button = locator
                        break
                    except Exception:
                        continue

                if all_data_button is None:
                    print(f"[ERROR] All Data option not found for {company}")
                    browser.close()
                    continue

                with page.expect_download(timeout=60000) as download_info:
                    all_data_button.click()

                download = download_info.value
                download.save_as(file_path)

                browser.close()

                print(f"Downloaded {company}: {file_path}")
                return file_path

        except PlaywrightTimeoutError as e:
            print(f"[TIMEOUT] {company}: {e}")
            time.sleep(3)

        except Exception as e:
            print(f"[ERROR] {company}: {e}")
            time.sleep(3)

    print(f"[FAILED] Could not download {company} after {retries} attempts.")
    return None


def download_all_companies(companies, headless: bool = False):
    downloaded_files = {}

    for company in companies:
        print("=" * 50)
        print(f"Downloading {company}")
        print("=" * 50)

        path = download_stock_data(company, headless=headless)

        downloaded_files[company] = path

    return downloaded_files


if __name__ == "__main__":
    COMPANIES = [
        "BBPL", "BCCL", "BFAL", "BIL", "BNBL",
        "BPCL", "BTCL", "DFAL", "DPL", "DPNB",
        "DWAL", "GICB", "KCL", "PCAL", "RICB",
        "STCB", "TBL"
    ]

    download_all_companies(COMPANIES, headless=False)