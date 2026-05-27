from scraper.downloader import download_all_companies

COMPANIES = [
    "BBPL", "BCCL", "BFAL", "BIL", "BNBL",
    "BPCL", "BTCL", "DFAL", "DPL", "DPNB",
    "DWAL", "GICB", "KCL", "PCAL", "RICB",
    "STCB", "TBL"
]

if __name__ == "__main__":

    download_all_companies(
        COMPANIES,
        headless=False
    )