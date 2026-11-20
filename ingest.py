"""
CBU Knowledge Base Ingestion
Chunks and embeds CBU content into ChromaDB.

Usage:
    python ingest.py --url https://www.calbaptist.edu/academics/
    python ingest.py --seed          # scrape all default CBU URLs
    python ingest.py --pdf ./docs/handbook.pdf
    python ingest.py --dir ./docs/     # .txt, .pdf, .docx, .xlsx
    python ingest.py --reset         # wipe collection and start fresh
"""

import argparse
import hashlib
import re
import time
from pathlib import Path

import chromadb
import ollama
import requests
from bs4 import BeautifulSoup

# --- Config ---
EMBED_MODEL = "nomic-embed-text"
# nomic-embed-text is trained with task prefixes; omitting them measurably
# degrades retrieval. Documents and queries must use different prefixes.
EMBED_DOC_PREFIX = "search_document: "
CHUNK_SIZE = 220           # words -- smaller chunks retrieve far more precisely
CHUNK_OVERLAP = 40
MIN_CHUNK_WORDS = 20
EMBED_MAX_CHARS = 6000      # comfortably inside nomic-embed-text's context
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001

CBU_SEED_URLS = [
    "https://www.calbaptist.edu/academics/",
    "https://www.calbaptist.edu/admissions/",
    "https://calbaptist.edu/life-at-cbu/",
    "https://www.calbaptist.edu/about/",
    "https://www.calbaptist.edu/engineering/",
    "https://calbaptist.edu/academics/programs/bachelor-of-science-computer-science/",
    "https://calbaptist.edu/academics/programs/minor-computer-science",
    "https://calbaptist.edu/academics/programs/minor-computer-engineering",
    "https://calbaptist.edu/academics/programs/master-of-science-computer-science/",
    "https://calbaptist.edu/academics/programs/minor-data-sciences",
    "https://calbaptist.edu/academics/programs/bachelor-of-science-data-science/",
    "https://calbaptist.edu/academics/programs/bachelor-of-science-electrical-and-computer-engineering/",
    "https://calbaptist.edu/academics/programs/minor-software-engineering-and-app-development"
]

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_or_create_collection(
    name="cbu_knowledge",
    metadata={"hnsw:space": "cosine"}
)


# Filenames abbreviate the things people actually say. A plan titled
# "..._ML_CONC" never contains the words "machine learning", so no amount of
# retrieval tuning finds it -- the title has to say what the document is.
_TITLE_EXPANSIONS = [
    ("ML CONC", "Machine Learning Concentration"),
    ("Cybersecurity CONC", "Cybersecurity Concentration"),
    ("General CONC", "General Concentration"),
    ("CONC", "Concentration"),
    ("CBU CS", "CBU Computer Science"),
    ("CoE", "College of Engineering"),
    ("BASAI", "BASAI Bachelor of Applied Science in Artificial Intelligence"),
    ("MSAI", "MSAI Master of Science in Artificial Intelligence"),
    ("GE", "GE General Education"),
    ("4 Year Plan", "four year course plan"),
    ("3 Year Program", "three year course plan"),
]


def _title_for(source: str) -> str:
    """Human-readable document name, used as a per-chunk context header."""
    name = Path(source).name if not source.startswith("http") else source
    for ext in (".txt", ".pdf", ".docx", ".xlsx"):
        name = name.replace(ext, "")
    name = name.replace("_", " ").replace("-", " ").strip()
    for short, long in _TITLE_EXPANSIONS:
        name = re.sub(rf"\b{re.escape(short)}\b", long, name)
    return re.sub(r"\s+", " ", name).strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, falling back to single newlines for flat text."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts or [text.strip()]


def chunk_text(text: str, source: str, title: str | None = None) -> list[dict]:
    """
    Chunk on paragraph boundaries, packing up to CHUNK_SIZE words.

    Splitting mid-sentence on a fixed word window scatters a single fact across
    two chunks and neither retrieves well. Every chunk also carries the document
    title so an isolated passage still says what it belongs to.
    """
    title = title or _title_for(source)
    paragraphs = _split_paragraphs(text)

    chunks: list[dict] = []
    buf: list[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words
        if buf_words >= MIN_CHUNK_WORDS:
            body = " ".join(buf).strip()
            chunks.append({
                "text": f"{title}\n\n{body}",
                "source": source,
                "title": title,
                "index": len(chunks),
            })
        buf, buf_words = [], 0

    for para in paragraphs:
        words = para.split()
        if buf_words and buf_words + len(words) > CHUNK_SIZE:
            tail = " ".join(buf).split()[-CHUNK_OVERLAP:]
            flush()
            buf, buf_words = [" ".join(tail)], len(tail)
        buf.append(para)
        buf_words += len(words)
        while buf_words > CHUNK_SIZE:  # a single oversized paragraph
            words_all = " ".join(buf).split()
            head, rest = words_all[:CHUNK_SIZE], words_all[CHUNK_SIZE - CHUNK_OVERLAP:]
            buf, buf_words = [" ".join(head)], len(head)
            flush()
            buf, buf_words = [" ".join(rest)], len(rest)
    flush()
    return chunks


def embed_and_store(chunks: list[dict]):
    for chunk in chunks:
        doc_id = hashlib.md5(chunk["text"].encode()).hexdigest()
        # The embedder has a fixed context; the stored text stays whole, only
        # what is embedded is capped. Word-count chunking cannot see a chunk
        # whose "words" are tables or runs of symbols.
        embedding = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=EMBED_DOC_PREFIX + chunk["text"][:EMBED_MAX_CHARS],
        )["embedding"]
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "source": chunk["source"],
                "title": chunk.get("title", ""),
                "index": chunk.get("index", 0),
            }],
        )
    print(f"  Stored {len(chunks)} chunks.")


def _fetch_html(url: str) -> str:
    """
    The page as text, decoded correctly.

    A server that sends text/html without a charset gets ISO-8859-1 from
    requests, per the old HTTP default -- and a UTF-8 page read that way stores
    "±" as "Â±" and every dash as "â". Prefer what the bytes say.
    """
    resp = requests.get(url, timeout=20, headers={"User-Agent": "CBU-Bot/1.0"})
    resp.raise_for_status()
    if "charset" not in resp.headers.get("content-type", "").lower():
        resp.encoding = resp.apparent_encoding or "utf-8"
        if resp.encoding.lower() in ("ascii", "iso-8859-1", "windows-1252"):
            resp.encoding = "utf-8"
    return resp.text


def _replace_source(url: str) -> None:
    """
    Drop what is already stored for this page before storing it again.

    Chunk IDs are hashes of the chunk text, so re-scraping a page whose text
    changed -- or was decoded differently -- added new chunks next to the old
    ones instead of replacing them.
    """
    try:
        collection.delete(where={"source": url})
    except Exception as e:
        print(f"  Could not clear old chunks for {url}: {e}")


def scrape_url(url: str, sections: bool = False, label: str | None = None):
    """
    Scrape one page into the collection, replacing what it stored before.

    With sections, every <section> becomes its own run of chunks headed
    "<label> -- <section heading>". A long single-page site chunked as one
    stream produces passages like "The leads: Christopher Petrov, President..."
    that never say whose leads they are, and a list of names with no owner
    outranks the real answer to "who is <someone>".
    """
    print(f"Scraping: {url}")
    try:
        soup = BeautifulSoup(_fetch_html(url), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "form", "noscript", "svg"]):
            tag.decompose()
        _replace_source(url)
        if not sections:
            text = soup.get_text(separator=" ", strip=True)
            chunks = chunk_text(text, source=url, title=label)
        else:
            page = label or (soup.title.get_text(strip=True) if soup.title
                             else _title_for(url))
            chunks = []
            heading = ""
            for section in soup.find_all("section"):
                h = section.find(["h1", "h2", "h3"])
                if h:
                    heading = h.get_text(" ", strip=True)
                body = section.get_text(separator=" ", strip=True)
                if len(body.split()) < MIN_CHUNK_WORDS:
                    continue
                title = f"{page} -- {heading}" if heading else page
                chunks.extend(chunk_text(body, source=url, title=title))
        embed_and_store(chunks)
        time.sleep(0.5)
    except Exception as e:
        print(f"  Error: {e}")


def ingest_pdf(pdf_path: str, label: str | None = None):
    try:
        import pymupdf
        print(f"Ingesting PDF: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        text = " ".join(page.get_text() for page in doc)
        # Table-of-contents dot leaders ("Cockpit ......... 55") are one
        # "word" hundreds of characters long, so a chunk packed by word count
        # blew past the embedding model's context and the whole PDF failed.
        text = re.sub(r"(?:\s?[.\u2026·]){4,}", " ", text)
        _replace_source(pdf_path)
        title = f"{label} -- {_title_for(pdf_path)}" if label else None
        chunks = chunk_text(text, source=pdf_path, title=title)
        embed_and_store(chunks)
    except ImportError:
        print("Install pymupdf: pip install pymupdf")


def ingest_docx(docx_path: str):
    try:
        import docx
        print(f"Ingesting DOCX: {docx_path}")
        document = docx.Document(docx_path)
        parts = [para.text for para in document.paragraphs if para.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        chunks = chunk_text("\n".join(parts), source=docx_path)
        embed_and_store(chunks)
    except ImportError:
        print("Install python-docx: pip install python-docx")


def ingest_xlsx(xlsx_path: str):
    try:
        import openpyxl
        print(f"Ingesting XLSX: {xlsx_path}")
        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        parts = []
        for sheet in wb.worksheets:
            parts.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        chunks = chunk_text("\n".join(parts), source=xlsx_path)
        embed_and_store(chunks)
    except ImportError:
        print("Install openpyxl: pip install openpyxl")


def ingest_directory(dir_path: str, skip_exts: tuple[str, ...] = ()):
    handlers = {".pdf": ingest_pdf, ".docx": ingest_docx, ".xlsx": ingest_xlsx}
    for path in sorted(Path(dir_path).rglob("*")):
        if not path.is_file() or path.name.startswith("~$") or path.name == ".DS_Store":
            continue
        if path.suffix in skip_exts:
            print(f"  Skipping {path.suffix} (converted separately): {path.name}")
            continue
        if path.suffix == ".txt":
            print(f"Ingesting: {path}")
            text = path.read_text(errors="ignore")
            embed_and_store(chunk_text(text, str(path)))
        elif path.suffix in handlers:
            handlers[path.suffix](str(path))
        else:
            print(f"  Skipping unsupported file: {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Single URL to scrape")
    parser.add_argument("--sections", action="store_true",
                        help="With --url: chunk each <section> on its own")
    parser.add_argument("--label",
                        help="With --url or --pdf: heading to put on every chunk")
    parser.add_argument("--seed", action="store_true", help="Scrape all seed URLs")
    parser.add_argument("--pdf", help="PDF to ingest")
    parser.add_argument("--dir", help="Directory of docs to ingest")
    parser.add_argument("--skip-ext", nargs="*", default=[],
                        help="Extensions to skip, e.g. --skip-ext .docx .xlsx")
    parser.add_argument("--reset", action="store_true", help="Wipe collection and exit")
    args = parser.parse_args()

    if args.reset:
        client.delete_collection("cbu_knowledge")
        print("Collection wiped.")
    elif args.url:
        scrape_url(args.url, sections=args.sections, label=args.label)
    elif args.seed:
        for url in CBU_SEED_URLS:
            scrape_url(url)
    elif args.pdf:
        ingest_pdf(args.pdf, label=args.label)
    elif args.dir:
        ingest_directory(args.dir, tuple(args.skip_ext))
    else:
        parser.print_help()

    print(f"Total docs in collection: {collection.count()}")
