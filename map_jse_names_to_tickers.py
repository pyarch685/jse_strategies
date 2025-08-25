
import re
import time
import json
import difflib
import requests
import pandas as pd
from typing import Optional, Tuple

YF_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def normalize_name(s: str) -> str:
    """Simplify names so 'Naspers Ltd' ~ 'Naspers'."""
    s = (s or "").lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)           # remove punctuation
    s = re.sub(r"\b(proprietary|pty|limited|ltd|plc|holdings?|group|corp(oration)?)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sim(a: str, b: str) -> float:
    """0..100 similarity using stdlib (no extra deps)."""
    return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()

def yahoo_search_jo(company_name: str, pause: float = 0.2) -> Tuple[Optional[str], float, str]:
    """
    Search Yahoo Finance, return (best_ticker, score, matched_name) for .JO tickers.
    """
    params = {"q": company_name, "quotesCount": 20, "newsCount": 0, "listsCount": 0}
    try:
        r = requests.get(YF_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        time.sleep(pause)
        return None, 0.0, ""

    query_norm = normalize_name(company_name)
    candidates = []
    for q in (data.get("quotes") or []):
        sym = q.get("symbol") or ""
        if not sym.endswith(".JO"):
            continue

        nm = (q.get("shortname") or q.get("longname") or q.get("name") or "").strip()
        nm_norm = normalize_name(nm)
        score = sim(query_norm, nm_norm) if nm_norm else 0.0

        base = re.sub(r"\.jo$", "", sym.lower())
        if base in query_norm or query_norm in base:
            score += 5

        candidates.append((score, sym, nm))

    time.sleep(pause)
    if not candidates:
        return None, 0.0, ""
    candidates.sort(reverse=True, key=lambda x: x[0])
    best = candidates[0]
    return best[1], best[0], best[2]

def map_file(
    companies_txt_path: str,
    out_codes_txt: str = "jse_codes.txt",
    out_mapping_csv: str = "company_to_ticker_map.csv",
    out_combined_txt: str = "companies_with_codes.txt",
    min_score: float = 60.0,
):
    # 1) Load names
    with open(companies_txt_path, "r", encoding="utf-8") as f:
        companies = [ln.strip() for ln in f if ln.strip()]

    results = []
    for name in companies:
        ticker, score, matched_name = yahoo_search_jo(name)
        if ticker is None or score < min_score:
            ticker = ""
        results.append((name, ticker, score, matched_name))

    # 2) Write codes.txt (aligned to input order)
    with open(out_codes_txt, "w", encoding="utf-8") as f:
        for _, ticker, _, _ in results:
            f.write((ticker or "") + "\n")

    # 3) Write combined mapping CSV
    df = pd.DataFrame(results, columns=["Company", "Ticker", "match_score", "matched_name"])
    df["needs_review"] = (df["Ticker"] == "") | (df["match_score"] < min_score)
    df.to_csv(out_mapping_csv, index=False)

    # 4) Also produce a combined TXT: "Company | Ticker"
    with open(out_combined_txt, "w", encoding="utf-8") as f:
        for name, ticker, _, _ in results:
            f.write(f"{name} | {ticker}\n")

    print(f"✅ Wrote {len(results)} tickers to {out_codes_txt}")
    print(f"✅ Wrote CSV to {out_mapping_csv}")
    print(f"✅ Wrote combined TXT to {out_combined_txt}")
    print("ℹ️  Review blank/low-score rows manually (needs_review in CSV).")

if __name__ == "__main__":
    # Change this path if needed
    map_file("jse_companies.txt")
