// Canned data for VITE_MOCK_CHAT=1, so the chat UI can be demoed without a backend.
// Loaded lazily, so it never ships in the main bundle of a live build.
import type {
  AgentAnswer,
  ChatEvent,
  ChatRequest,
  Chunk,
  CompaniesResponse,
  ModelsResponse,
} from "./types";

export const MOCK_MODELS: ModelsResponse = {
  models: [
    { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash", provider: "google" },
    { id: "llama-3.3-70b-versatile", label: "Llama 3.3 70B", provider: "groq" },
    { id: "qwen3:8b", label: "Qwen3 8B (local)", provider: "ollama" },
  ],
  default: "gemini-2.5-flash",
};

export const MOCK_COMPANIES: CompaniesResponse = {
  companies: [
    { ticker: "AAPL", name: "Apple Inc.", sector: "Information Technology" },
    { ticker: "MSFT", name: "Microsoft Corporation", sector: "Information Technology" },
    { ticker: "NVDA", name: "NVIDIA Corporation", sector: "Information Technology" },
    { ticker: "AMZN", name: "Amazon.com, Inc.", sector: "Consumer Discretionary" },
    { ticker: "GOOGL", name: "Alphabet Inc.", sector: "Communication Services" },
    { ticker: "JPM", name: "JPMorgan Chase & Co.", sector: "Financials" },
    { ticker: "JNJ", name: "Johnson & Johnson", sector: "Health Care" },
    { ticker: "XOM", name: "Exxon Mobil Corporation", sector: "Energy" },
    { ticker: "KO", name: "The Coca-Cola Company", sector: "Consumer Staples" },
    { ticker: "CAT", name: "Caterpillar Inc.", sector: "Industrials" },
  ],
};

const FILING = "AAPL-10K-2024FY";

const CHUNKS: Chunk[] = [
  {
    chunk_id: `${FILING}:section:00142`,
    filing_id: FILING,
    ticker: "AAPL",
    company: "Apple Inc.",
    form: "10-K",
    fiscal_year: 2024,
    fiscal_period: "FY",
    doc_name: null,
    strategy: "section",
    section: "Item 7. MD&A > Segment Operating Performance > Products and Services Performance",
    page: 22,
    page_end: 23,
    text: [
      "The following table shows net sales by category for 2024, 2023 and 2022 (dollars in millions):",
      "",
      "| Category | 2024 | Change | 2023 | Change | 2022 |",
      "| --- | ---: | ---: | ---: | ---: | ---: |",
      "| iPhone | $ 201,183 | — % | $ 200,583 | (2)% | $ 205,489 |",
      "| Mac | 29,984 | 2 % | 29,357 | (27)% | 40,177 |",
      "| iPad | 26,694 | (6)% | 28,300 | (3)% | 29,292 |",
      "| Wearables, Home and Accessories | 37,005 | (7)% | 39,845 | (3)% | 41,241 |",
      "| Services | 96,169 | 13 % | 85,200 | 9 % | 78,129 |",
      "| Total net sales | $ 391,035 | 2 % | $ 383,285 | (3)% | $ 394,328 |",
      "",
      "Total net sales increased 2% or $7.8 billion during 2024 compared to 2023.",
    ].join("\n"),
    has_table: true,
    table_ids: [`${FILING}:t0031`],
  },
  {
    chunk_id: `${FILING}:section:00145`,
    filing_id: FILING,
    ticker: "AAPL",
    company: "Apple Inc.",
    form: "10-K",
    fiscal_year: 2024,
    fiscal_period: "FY",
    doc_name: null,
    strategy: "section",
    section: "Item 7. MD&A > Segment Operating Performance > Services",
    page: 23,
    page_end: 23,
    text:
      "Services net sales increased during 2024 compared to 2023 primarily due to higher net sales " +
      "from advertising, the App Store and cloud services. Services net sales were $96.2 billion in 2024, " +
      "compared to $85.2 billion in 2023, and represented 25% of total net sales.",
    has_table: false,
    table_ids: [],
  },
  {
    chunk_id: `${FILING}:xbrl:Revenues`,
    filing_id: FILING,
    ticker: "AAPL",
    company: "Apple Inc.",
    form: "10-K",
    fiscal_year: 2024,
    fiscal_period: "FY",
    doc_name: null,
    strategy: "section",
    section: "XBRL company facts > us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    page: null,
    page_end: null,
    text: [
      "| Concept | Period | Value (USD) | Form | Filed |",
      "| --- | --- | ---: | --- | --- |",
      "| RevenueFromContractWithCustomerExcludingAssessedTax | FY2024 (ended 2024-09-28) | 391,035,000,000 | 10-K | 2024-11-01 |",
      "| RevenueFromContractWithCustomerExcludingAssessedTax | FY2023 (ended 2023-09-30) | 383,285,000,000 | 10-K | 2023-11-03 |",
    ].join("\n"),
    has_table: true,
    table_ids: [],
  },
];

export function getMockChunk(chunkId: string): Chunk {
  const chunk = CHUNKS.find((c) => c.chunk_id === chunkId);
  if (!chunk) throw new Error(`Unknown chunk ${chunkId}`);
  return chunk;
}

const ANSWER_TEXT =
  "Apple's total net sales were **$391.0 billion** in fiscal 2024, up 2% from $383.3 billion in fiscal 2023 [1].\n\n" +
  "- **Services** grew fastest, reaching $96.2 billion versus $85.2 billion a year earlier [2].\n" +
  "- **iPhone** was roughly flat at $201.2 billion, still about half of total sales [1].\n\n" +
  "The XBRL company facts confirm total revenue of $391,035 million for the year ended September 28, 2024 [3].";

const ANSWER: AgentAnswer = {
  answer: ANSWER_TEXT,
  refused: false,
  answer_value: 391035000000,
  answer_unit: "USD",
  citations: [
    {
      chunk_id: CHUNKS[0]!.chunk_id,
      filing_id: FILING,
      section: CHUNKS[0]!.section,
      page: 22,
      page_end: 23,
      quote: "Total net sales increased 2% or $7.8 billion during 2024 compared to 2023.",
    },
    {
      chunk_id: CHUNKS[1]!.chunk_id,
      filing_id: FILING,
      section: CHUNKS[1]!.section,
      page: 23,
      quote: "Services net sales were $96.2 billion in 2024, compared to $85.2 billion in 2023",
    },
    {
      chunk_id: CHUNKS[2]!.chunk_id,
      filing_id: FILING,
      section: CHUNKS[2]!.section,
      page: null,
      quote: "391,035,000,000",
    },
  ],
  retrieved_chunk_ids: CHUNKS.map((c) => c.chunk_id),
  steps: ["plan", "search_filings", "get_company_facts", "verify", "answer", "self_check"],
  usage: {
    prompt_tokens: 8214,
    completion_tokens: 412,
    llm_calls: 4,
    cost_usd: 0.00289,
    latency_ms: 5840,
  },
};

const REFUSAL: AgentAnswer = {
  answer:
    "I could not find support for this in the indexed filings, so I am not answering. " +
    "Try naming the company and fiscal year, or ask about a figure reported in a 10-K or 10-Q.",
  refused: true,
  answer_value: null,
  answer_unit: null,
  citations: [],
  retrieved_chunk_ids: [],
  steps: ["plan", "search_filings", "verify", "answer"],
  usage: { prompt_tokens: 3120, completion_tokens: 96, llm_calls: 2, cost_usd: 0.00098, latency_ms: 2210 },
};

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

/** Replays the same event sequence the backend sends, with realistic pacing. */
export async function replayMockChat(
  request: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  // Questions about something other than revenue get a refusal, to demo that path too.
  const refuse = !/revenue|sales|apple|aapl/i.test(request.question);
  const answer = refuse ? REFUSAL : ANSWER;
  const steps: Array<[string, string]> = refuse
    ? [
        ["plan", "Identify the company, metric and period"],
        ["search_filings", "hybrid search, top 10, no filters"],
        ["verify", "No retrieved passage supports an answer"],
        ["answer", "Writing a refusal"],
      ]
    : [
        ["plan", "Find FY2024 total net sales for AAPL and the FY2023 comparison"],
        ["search_filings", "\"net sales by category 2024\" (AAPL, 10-K, FY2024): 8 chunks"],
        ["get_company_facts", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"],
        ["verify", "Filing table matches XBRL fact within 0.0%"],
        ["answer", "Drafting answer with 3 citations"],
      ];

  for (const [name, detail] of steps) {
    onEvent({ type: "step", data: { name, detail } });
    await sleep(name === "search_filings" ? 700 : 450, signal);
  }
  const words = answer.answer.match(/\S+\s*/g) ?? [];
  for (const word of words) {
    onEvent({ type: "token", data: { text: word } });
    await sleep(28, signal);
  }
  if (!refuse) {
    onEvent({ type: "step", data: { name: "self_check", detail: "Every number has a citation" } });
    await sleep(400, signal);
  }
  onEvent({ type: "answer", data: answer });
  onEvent({ type: "done", data: {} });
}
