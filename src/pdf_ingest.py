# ============================================================
#  src/pdf_ingest.py  —  PDF source (page-by-page chunking)
#
#  Reads every PDF in data/pdfs/ and turns it into retrievable
#  documents: each PAGE is parsed, cleaned, and chunked, and
#  every chunk is tagged with its filename + page number so the
#  CEO agent can cite evidence precisely ("Annual Report p.14").
#
#  Drop PDFs (annual report, investor deck, whitepapers) into
#  data/pdfs/.  Optionally list public PDF URLs in config.PDF_URLS
#  to have them downloaded automatically.
#
#  Run (with the rest): python -m src.collect
# ============================================================

import os
import glob
import json

import requests
from pypdf import PdfReader

from . import utils
import config as cfg


# --- 1. Optional manifest: nicer titles/URLs for citations ---
def _manifest():
    """data/pdfs/sources.json maps  filename -> {"title":..,"url":..} (optional)."""
    path = os.path.join(cfg.PDF_DIR, "sources.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# --- 2. Optional auto-download of public PDFs ----------------
def download_pdfs():
    """Fetch PDFs listed in config.PDF_URLS (dict: filename -> url)."""
    os.makedirs(cfg.PDF_DIR, exist_ok=True)
    urls = getattr(cfg, "PDF_URLS", {})
    items = urls.items() if isinstance(urls, dict) else [(None, u) for u in urls]
    for name, url in items:
        # Use the given filename, else derive one from the URL tail.
        fname = name or (url.rstrip("/").split("/")[-1].split("?")[0] or "download.pdf")
        if not fname.lower().endswith(".pdf"):
            fname += ".pdf"
        dest = os.path.join(cfg.PDF_DIR, fname)
        if os.path.exists(dest):
            continue
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (research)"})
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            print(f"        downloaded {fname}")
        except Exception as e:
            print(f"        ! could not download {url} ({e})")


# --- 3. Parse every PDF, page by page ------------------------
def collect_pdfs():
    print("  [PDF] parsing local + official investor PDFs ...")
    download_pdfs()                                   # no-op if PDF_URLS is empty
    docs = []
    manifest = _manifest()

    # Scan BOTH the manual PDF folder and the official investor-PDF folder
    # (the CDN downloader in collect.py fills the latter).
    dirs = [cfg.PDF_DIR, getattr(cfg, "INVESTOR_PDF_DIR", "")]
    pdfs = []
    for d in dirs:
        if d:
            pdfs += sorted(glob.glob(os.path.join(d, "*.pdf")))
    if not pdfs:
        print(f"        (no PDFs found in {dirs})")
        return docs

    for path in pdfs:
        fname = os.path.basename(path)
        meta  = manifest.get(fname, {})
        title = meta.get("title", fname.replace(".pdf", ""))
        url   = meta.get("url", path)

        try:
            reader = PdfReader(path)
        except Exception as e:
            print(f"        ! {fname} unreadable ({e})")
            continue

        chars = 0
        page_texts = []
        for page in reader.pages:
            t = utils.clean_text(page.extract_text() or "")
            if len(t) >= 30:
                page_texts.append(t)
        full = " ".join(page_texts)
        if len(full) < 100:
            print(f"        ! {fname}: no extractable text (scanned PDF?)")
            continue
        docs.append(utils.make_doc(                # ONE full-text doc per PDF, no chunking
            title=title, body=full, source="pdf", url=url, section=fname,
        ))
        print(f"        -> {fname}: {len(full):,} chars from {len(reader.pages)} pages")

    print(f"        -> {len(docs)} full PDFs")
    return docs
