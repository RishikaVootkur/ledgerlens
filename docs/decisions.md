# Design decisions

Each entry: the decision, the reason, and what was rejected. Newest last.

## D1. Python 3.12 with uv, src layout, strict mypy
uv gives a locked, reproducible environment that CI can install with `--frozen`. Strict mypy catches the shape errors that are common in eval code (dicts of metrics passed around). Python 3.12 because several observability packages cap below 3.15 and the brief rules out the system 3.14.

## D2. Gemini through the OpenAI-compatible endpoint
On this account the native `generateContent` endpoint returns 404 for current models, while `https://generativelanguage.googleapis.com/v1beta/openai/` serves them. LiteLLM talks to it as an OpenAI-compatible provider, so every model still goes through one provider layer.

## D3. Model roles
- Answer model (default): `gemini-3.8-flash`, the newest Gemini model with free quota on this key.
- Judge: `openai/gpt-oss-120b` on Groq, a different model family from every answer model under test (Gemini, Qwen), so the judge is never grading its own family.
- Comparison models: `gemini-3.8-flash`, `qwen/qwen3.8-27b` on Groq, local `qwen3:8b` on Ollama.
- Groq's 8000 tokens per minute cap means agent prompts must stay compact; the judge prompt is short, which is why the judge lives on Groq.

## D4. JSON-file response cache keyed on the full request
Every LLM request (model, messages, tools, sampling parameters, provider extras) is hashed with SHA-256 and the response is stored as one JSON file. Files instead of SQLite so the slice the CI gate replays can be committed and read in a diff. `LLM_MODE=replay` serves only from the cache and raises `CacheMissError` on a miss, which is what makes the CI gate deterministic, free and keyless. Replays report the latency recorded live, so latency metrics do not collapse to zero on replay.

## D5. List-price-equivalent cost
Each model carries its public list price per million tokens (checked 2026-11-01): Gemini 3.8 Flash $0.75 in / $3.75 out (Google's introductory rate through 2026-12-31, per several pricing trackers; Google's own page is ai.google.dev/gemini-api/docs/pricing), GPT-OSS 120B on Groq $0.15 / $0.60, Qwen3.8 27B on Groq $0.80 / $4.00 (preview rate). Local qwen3:8b has no list price and is priced at $0. Gemini's OpenAI-compatible endpoint reports thinking tokens only in `total_tokens`, so billable output is `total - prompt`; ignoring that would under-count Gemini cost by an order of magnitude on short answers.

## D6. Retries without silent fallback
Retries use exponential backoff with jitter (up to 8 attempts, 90 s cap) on rate limits, 5xx and connection errors, never on auth or bad-request errors. There is deliberately no automatic fallback to another model: a fallback would silently mix models inside one run and corrupt model comparisons. Client-side pacing keeps each model under its free-tier requests-per-minute.

## D7. Retrieval layout: SQLite for text, Qdrant for vectors, bm25s for keywords
Chunk text and metadata live in one SQLite file keyed by chunk id; Qdrant stores only vectors plus the filterable fields (ticker, year, form, filing, doc), and BM25 stores ids plus a metadata array. Every consumer that needs a passage (citations, `get_table`, the UI) reads it from SQLite, so the API container never holds the corpus in memory. BM25 filtering scores the whole corpus and masks it rather than filtering a top-N, so narrow filters never come back short. Hybrid uses reciprocal rank fusion (k = 60, the value from Cormack et al. 2009) over the top 40 of each retriever; the optional rerank is `Xenova/ms-marco-MiniLM-L-6-v2` through fastembed's ONNX runtime, which avoids pulling torch into the image. Embedding is resumable: a chunk is re-embedded only when its text hash changes. Measured embedding throughput with `nomic-embed-text` on the M-series host: about 19 chunks per second for ~500-token chunks.
