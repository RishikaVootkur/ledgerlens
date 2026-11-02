# LedgerLens web

React + TypeScript + Vite + Tailwind UI for LedgerLens: the **Ask** page (streamed, cited answers
from SEC filings) and the **Eval lab** (headline metrics with 95% CIs, ablations, model comparison,
failure taxonomy, judge calibration, per-question run detail).

Needs Node 22.12 or newer. All tooling is project-local.

## Commands

```bash
cd web
npm ci                 # install from package-lock.json
npm run dev            # dev server on :5173, proxies /api to http://localhost:8000
npm run dev:mock       # no backend needed: sample eval data + replayed chat stream
npm run lint           # eslint
npm test               # vitest (SSE parser, citation markers, metric formatting, components)
npm run build          # tsc type-check + production build into web/dist
npm run build:static   # GitHub Pages build (no chat, base /ledgerlens/, data from ./data)
npm run preview        # serve the last build
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE` | `/api` | Where the FastAPI routes live. |
| `VITE_API_PROXY` | `http://localhost:8000` | Dev server only: target for the `/api` proxy. |
| `VITE_DATA_BASE` | `/api/data` | Folder holding `dashboard.json` and `runs/{run_id}.json`. Use `/mock` for the sample files in `public/mock`. |
| `VITE_STATIC` | unset | `1` turns off chat (static hosting with no backend). |
| `VITE_MOCK_CHAT` | unset | `1` replays a recorded SSE sequence instead of calling `/api/chat`. |
| `VITE_BASE` | `/` | Vite `base`; GitHub Pages serves the site under `/ledgerlens/`. |

Routing uses the URL hash (`#/ask`, `#/eval`, `#/runs/{run_id}?q={qid}`), so deep links work on
GitHub Pages without server rewrites. In mock chat mode, `#/ask?q=...` also runs the question, which
makes README screenshots reproducible.

## Publishing the static dashboard

`npm run build:static` writes `web/dist`. Copy `dashboard.json` and `runs/*.json` from the eval
lab into `web/dist/data/` before uploading the folder to Pages. The sample files in `public/mock`
are shipped too, but nothing reads them unless `VITE_DATA_BASE=/mock`.

## Backend contract

Types live in `src/api/types.ts` and mirror `src/ledgerlens/schemas.py`; the fetch and SSE code is
in `src/api/client.ts`.

- `POST /api/chat` with `{question, filters?: {ticker?, fiscal_year?, form?}, model?}` returns
  `text/event-stream` with events `step` `{name, detail}`, `token` `{text}`, `answer` (AgentAnswer),
  `error` `{message}` and `done` `{}`. Inline markers `[1]`, `[2]` in the answer refer to
  `answer.citations` by 1-based position.
- `GET /api/chunks/{chunk_id}` returns a Chunk. The id is URL-encoded (`:` becomes `%3A`).
- `GET /api/models` returns `{models: [{id, label, provider}], default}`.
- `GET /api/companies` returns `{companies: [{ticker, name, sector}]}`.
- `GET {DATA_BASE}/dashboard.json` (type `Dashboard`) and `GET {DATA_BASE}/runs/{run_id}.json`
  (type `RunFile`).
