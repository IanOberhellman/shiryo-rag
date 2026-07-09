"""Shared RAG pipeline: retrieval from Chroma + grounded answering via Claude."""

import json
import os

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION = "tuj_handbooks"

SYSTEM_PROMPT = (
    "You answer questions about Temple University Japan (TUJ) using ONLY the handbook "
    "excerpts provided in the user message. If the excerpts do not contain the information "
    "needed, set answerable=false and say the handbooks don't cover it — never guess and "
    "never use outside knowledge. When you answer, cite the chunk ids you relied on, each "
    "with a short verbatim quote from that excerpt supporting the answer."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": "true only if the provided excerpts contain the information needed to answer",
        },
        "answer": {
            "type": "string",
            "description": "The answer grounded in the excerpts. If not answerable, a short statement that the handbooks don't cover this.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "quote": {
                        "type": "string",
                        "description": "Short verbatim quote from that chunk supporting the answer",
                    },
                },
                "required": ["chunk_id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answerable", "answer", "citations"],
    "additionalProperties": False,
}


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION)


def retrieve(question: str, k: int = 8) -> list[dict]:
    col = get_collection()
    res = col.query(query_texts=[question], n_results=k)
    chunks = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        chunks.append(
            {
                "id": res["ids"][0][i],
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
            }
        )
    return chunks


def answer(question: str, k: int = 8) -> dict:
    chunks = retrieve(question, k)
    context = "\n\n".join(
        f"[{c['id']}] (from {c['source']}, page {c['page']})\n{c['text']}"
        for c in chunks
    )
    client = Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
        messages=[
            {
                "role": "user",
                "content": f"Handbook excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    result = json.loads(text)
    result["chunks"] = chunks
    result["usage"] = {
        "input": resp.usage.input_tokens,
        "output": resp.usage.output_tokens,
    }
    return result
