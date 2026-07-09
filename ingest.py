"""Ingest PDFs from docs/ into a local Chroma vector store, one chunk per page
(long pages split in half). Run once before using the app: python ingest.py"""

import os

import chromadb
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION = "tuj_handbooks"
MAX_CHUNK_CHARS = 2200


def split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    mid = len(text) // 2
    split_at = text.rfind("\n", 0, mid)
    if split_at < 200:
        split_at = mid
    return split_long(text[:split_at]) + split_long(text[split_at:])


def main():
    pdfs = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf"))
    if not pdfs:
        raise SystemExit("No PDFs in docs/ — see README for download instructions.")

    ids, texts, metas = [], [], []
    for pdf in pdfs:
        short = pdf.replace(".pdf", "").replace("tuj-", "")[:30]
        reader = PdfReader(os.path.join(DOCS_DIR, pdf))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < 80:  # skip covers / near-empty pages
                continue
            for part_num, part in enumerate(split_long(text)):
                suffix = f"-{chr(97 + part_num)}" if len(split_long(text)) > 1 else ""
                ids.append(f"{short}-p{page_num}{suffix}")
                texts.append(part)
                metas.append({"source": pdf, "page": page_num})

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    # add in batches — the first run downloads the local embedding model
    BATCH = 50
    for i in range(0, len(ids), BATCH):
        col.add(ids=ids[i : i + BATCH], documents=texts[i : i + BATCH], metadatas=metas[i : i + BATCH])
        print(f"embedded {min(i + BATCH, len(ids))}/{len(ids)} chunks")

    print(f"Done: {len(ids)} chunks from {len(pdfs)} PDFs -> {DB_DIR}")


if __name__ == "__main__":
    main()
