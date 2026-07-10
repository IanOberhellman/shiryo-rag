# 📚 Shiryō RAG (資料RAG)

**Ask the Temple University Japan student handbooks anything — every answer grounded in the documents, with citations, and an evaluation harness that proves it doesn't make things up.**

## Why this exists

The #1 concern Japanese companies report about generative AI is **accuracy** (50.4% of firms, Yomiuri/Teikoku Databank 2026 survey). A chatbot that confidently invents answers is worse than no chatbot. This project demonstrates the standard answer to that fear: **RAG (Retrieval-Augmented Generation) with an evaluation layer** — the model may only answer from retrieved document excerpts, must cite its sources, must refuse when the documents don't contain the answer, and a repeatable exam *measures* whether that actually holds.

## Eval results — including the bug the eval caught

The exam: 15 questions against the two official TUJ student handbooks — 12 with known answers in the documents, 3 "trick" questions whose answers are *not* in the documents (the correct behavior is refusal).

| Run | Score | Hallucinations on trick questions |
|---|---|---|
| v1 (retrieval k=5) | 14/15 (93%) | 0/3 |
| v2 (retrieval k=8) | 15/15 (100%) | 0/3 |
| v3 (+1 document, +1 question) | **16/16 (100%)** | **0/3** |

The v1 failure was diagnostic gold: "How long does the work permit application process take?" was wrongly refused. Investigation showed the chunk containing the answer ranked **7th** in retrieval while the system only fetched the top 5 — the knowledge was indexed but never reached the model. Raising retrieval depth to k=8 fixed it without regressing any other question. Full per-question results: [eval_results.md](eval_results.md).

This is why the eval exists: without it, that failure would have been invisible.

**v3 demonstrates RAG's operational superpower — updating knowledge by filing a document.** During testing, "Does TUJ have an AI major?" was (correctly) refused: the student-life handbooks don't mention the brand-new Fall 2026 B.S. in Artificial Intelligence. Adding TUJ's [official announcement](https://en-news.tuj.ac.jp/2026/04/03/bachelor-of-science-in-artificial-intelligence/) as one text file and re-running `ingest.py` flipped the answer to *"Yes, launching Fall 2026"* — with a citation. No retraining, no code changes: updating the AI's knowledge is a filing task.

## Architecture

```
PDFs (docs/) → ingest.py → chunks + local embeddings → ChromaDB (on disk)
                                                            ↓
question → retrieve top-8 chunks → Claude Sonnet 5 → JSON: {answerable, answer, citations[]}
                                   (may ONLY use          ↓
                                    the excerpts)    Streamlit UI with quoted sources
```

- **Embeddings & vector store:** ChromaDB with its local ONNX MiniLM model — free, runs on-device, no external services.
- **Generation:** Claude Sonnet 5 via the Anthropic API, with **structured outputs** (JSON schema) so the app can always parse the answer, the citations, and the `answerable` flag.
- **Grounding:** the system prompt forbids outside knowledge; `answerable=false` is a first-class outcome shown honestly in the UI.
- **Eval harness (`eval.py`):** fixed question set with expected keywords + expected refusals, keyword-based grading, results written to `eval_results.md`. Rerunnable after every change — regression testing for an AI system.

## Run it yourself

```bash
git clone https://github.com/IanOberhellman/shiryo-rag
cd shiryo-rag
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# The TUJ handbooks are not committed (they're TUJ's documents) — download them:
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
curl -sL -A "$UA" -o docs/tuj-student-handbook-fall-2025.pdf "https://storage.tuj.ac.jp/pdf/ug-osse/tuj-all-student-handbook-fall-2025_v2.pdf"
curl -sL -A "$UA" -o docs/tuj-international-student-handbook-spring-2026.pdf "https://storage.tuj.ac.jp/pdf/ug-osse/tuj-international-student-handbook-spring-2026.pdf"

./venv/bin/python ingest.py     # build the local index (~1 min first run)
./venv/bin/streamlit run app.py # ask questions
./venv/bin/python eval.py       # run the exam yourself
```

Works with any PDFs — drop your own documents in `docs/`, re-run `ingest.py`, and write your own `eval_set.json`.

## AI-assistance disclosure

Built by directing Claude (Anthropic) as a pair programmer: I specified the product, made the design decisions, and validated the results; Claude wrote the implementation. The eval failure analysis above happened live during development — the diagnosis (retrieval depth, not model quality) is the part I consider the real work.

## Limitations

- Keyword-based grading is a v1 evaluation method — it can't judge answer *quality*, only factual presence. Planned upgrade: LLM-judged faithfulness scoring (Ragas-style) on a larger question set.
- English documents only so far; Japanese-document retrieval (different tokenization) is the natural next step and a genuinely harder problem.
- 15 questions is a smoke test, not a benchmark. More questions → more trustworthy score.
- Handbooks change; answers are only as current as the PDFs in `docs/`.
