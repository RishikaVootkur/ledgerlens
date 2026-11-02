# Progress log

Newest entry first. Updated at the end of every PR so the build can resume from here.

## Status

- Done: M0 skeleton (in review)
- In progress: M0
- Next: M1 LLM provider layer, then the thin end-to-end slice (M2)
- Blockers: none

## Environment facts (checked 2026-11-01)

- Gemini free tier on this key: `gemini-3.8-flash` and `gemini-3.5-flash-lite` answer. `gemini-4-argon` and `gemini-3.1-pro-preview` have zero free quota. The native `generateContent` endpoint returns 404 for new models, so Gemini is called through Google's OpenAI-compatible endpoint.
- Groq free tier: `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`. Limits: 1000 requests/day, 8000 tokens/minute per model.
- Ollama: `qwen3:8b`, `nomic-embed-text`.
- Docker Desktop has 8 GB for containers.

## Key numbers

None yet.

## Log

### 2026-11-01
- Created the public repo with the scaffold commit. Started `chore/project-skeleton`.
