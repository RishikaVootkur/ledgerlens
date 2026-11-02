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
