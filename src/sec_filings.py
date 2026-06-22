# ============================================================
#  src/sec_filings.py  —  SEC EDGAR: filings + earnings (2y)
#
#  Pulls from the authoritative, reliable, no-JS source:
#    - 10-K  (annual report)      full text
#    - 10-Q  (quarterly report)   full text
#    - 8-K earnings (item 2.02)   -> press release (Ex 99.1)
#                                    + CFO commentary (Ex 99.2)
#  All stored as full-text docs (no chunking — done later).
# ============================================================

import time
from datetime import date, timedelta

import requests

from . import utils
import config as cfg

HEADERS = {"User-Agent": cfg.SEC_USER_AGENT}
CIK_N = str(int(cfg.CIK))


def _get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(0.2)                        # stay under SEC's 10 req/sec
    return r


def _accn_nodash(accn):
    return accn.replace("-", "")


def _file_url(accn, name):
    return f"https://www.sec.gov/Archives/edgar/data/{CIK_N}/{_accn_nodash(accn)}/{name}"


def _full_text(accn, name):
    return utils.clean_text(_get(_file_url(accn, name)).text)


def _htm_exhibits(accn):
    """List the .htm files in a filing (so we can find the press release)."""
    url = f"https://www.sec.gov/Archives/edgar/data/{CIK_N}/{_accn_nodash(accn)}/index.json"
    items = _get(url).json()["directory"]["item"]
    return [it["name"] for it in items if it["name"].endswith(".htm")]


# --- 1. 10-K / 10-Q + earnings 8-K in the last N years ------
def recent_filings():
    url = f"https://data.sec.gov/submissions/CIK{cfg.CIK}.json"
    recent = _get(url).json()["filings"]["recent"]
    cutoff = (date.today() - timedelta(days=365 * cfg.SEC_YEARS)).isoformat()
    items_list = recent.get("items", [""] * len(recent["form"]))
    out = []
    for form, accn, doc, fdate, items in zip(
        recent["form"], recent["accessionNumber"],
        recent["primaryDocument"], recent["filingDate"], items_list,
    ):
        if fdate < cutoff:
            continue
        if form in ("10-K", "10-Q"):
            out.append({"form": form, "accession": accn, "doc": doc, "date": fdate, "earnings": False})
        elif form == "8-K" and "2.02" in (items or ""):     # 2.02 = Results of Operations
            out.append({"form": form, "accession": accn, "doc": doc, "date": fdate, "earnings": True})
    return out


# --- 2. Collect every report/press release as full text -----
def collect_filings():
    print(f"  [SEC] EDGAR filings + earnings (last {cfg.SEC_YEARS}y) ...")
    docs = []
    try:
        filings = recent_filings()
    except Exception as e:
        print(f"        ! submissions failed ({e})")
        return docs

    for f in filings:
        try:
            if f["earnings"]:
                # 8-K earnings: grab the press release + CFO commentary exhibits
                names = _htm_exhibits(f["accession"])
                wanted = [n for n in names
                          if ("pr" in n.lower() or "commentary" in n.lower())
                          and not n.lower().startswith("nvda-")]
                if not wanted:
                    wanted = [f["doc"]]
                for n in wanted:
                    text = _full_text(f["accession"], n)
                    if len(text) < 300:
                        continue
                    label = "CFO commentary" if "commentary" in n.lower() else "earnings press release"
                    docs.append(utils.make_doc(
                        title=f"NVIDIA {label} ({f['date']})", body=text, source="filing",
                        url=_file_url(f["accession"], n), published=f["date"],
                        section=f"8-K {label} {f['date']}"))
                    print(f"        -> 8-K {label} {f['date']}: {len(text):,} chars")
            else:
                text = _full_text(f["accession"], f["doc"])
                if len(text) < 500:
                    continue
                kind = "Annual report" if f["form"] == "10-K" else "Quarterly report"
                docs.append(utils.make_doc(
                    title=f"NVIDIA {f['form']} {kind} ({f['date']})", body=text, source="filing",
                    url=_file_url(f["accession"], f["doc"]), published=f["date"],
                    section=f"{f['form']} {f['date']}"))
                print(f"        -> {f['form']} {f['date']}: {len(text):,} chars")
        except Exception as e:
            print(f"        ! {f['form']} {f['date']} failed ({e})")
            continue

    print(f"        -> {len(docs)} filing/earnings docs")
    return docs


# --- 3. Clean XBRL financials (dashboard) -------------------
def financials():
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cfg.CIK}.json"
    try:
        facts = _get(url).json()["facts"]["us-gaap"]
    except Exception as e:
        print(f"        ! financials skipped ({e})")
        return {}

    def latest_annual(concepts):
        for c in concepts:
            if c in facts:
                vals = [v for v in facts[c]["units"].get("USD", []) if v.get("form") == "10-K"]
                if vals:
                    return sorted(vals, key=lambda v: v["end"])[-1]["val"]
        return None

    return {
        "revenue":    latest_annual(["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"]),
        "net_income": latest_annual(["NetIncomeLoss"]),
        "assets":     latest_annual(["Assets"]),
    }
