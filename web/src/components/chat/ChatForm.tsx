import { useId, useState, type FormEvent, type KeyboardEvent } from "react";
import type { ChatRequest, CompanyInfo, ModelInfo } from "../../api/types";
import { Spinner } from "../ui";
import { buttonClass, inputClass } from "../../lib/style";

const CURRENT_YEAR = new Date().getFullYear();
const YEARS = Array.from({ length: 10 }, (_, i) => CURRENT_YEAR - i);
const FORMS = ["10-K", "10-Q"];

const EXAMPLE_QUESTIONS = [
  "What were Apple's total net sales in fiscal 2024, and how did Services grow?",
  "How much did JPMorgan set aside for credit losses in 2024 compared to 2023?",
  "What was Coca-Cola's operating margin in fiscal 2023?",
];

export function ChatForm({
  companies,
  models,
  defaultModel,
  busy,
  onSubmit,
  onCancel,
  initialQuestion = "",
}: {
  initialQuestion?: string;
  companies: CompanyInfo[];
  models: ModelInfo[];
  defaultModel: string | null;
  busy: boolean;
  onSubmit: (request: ChatRequest) => void;
  onCancel: () => void;
}) {
  const ids = { q: useId(), co: useId(), yr: useId(), form: useId(), model: useId() };
  const [question, setQuestion] = useState(initialQuestion);
  const [ticker, setTicker] = useState("");
  const [year, setYear] = useState("");
  const [form, setForm] = useState("");
  const [model, setModel] = useState<string | null>(null);
  const selectedModel = model ?? defaultModel ?? "";

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    const filters: NonNullable<ChatRequest["filters"]> = {};
    if (ticker) filters.ticker = ticker;
    if (year) filters.fiscal_year = Number(year);
    if (form) filters.form = form;
    onSubmit({
      question: q,
      ...(Object.keys(filters).length > 0 ? { filters } : {}),
      ...(selectedModel ? { model: selectedModel } : {}),
    });
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const selectClass = `${inputClass} pr-8`;
  const labelClass = "mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400";

  return (
    <form onSubmit={submit} className="space-y-3">
      <div>
        <label htmlFor={ids.q} className="sr-only">
          Question
        </label>
        <textarea
          id={ids.q}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          rows={3}
          placeholder="Ask about a US public company's 10-K or 10-Q…"
          className={`${inputClass} resize-y text-[0.9375rem] leading-relaxed`}
        />
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Enter to ask, Shift+Enter for a new line.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="col-span-2 sm:col-span-1">
          <label htmlFor={ids.co} className={labelClass}>
            Company
          </label>
          <select id={ids.co} value={ticker} onChange={(e) => setTicker(e.target.value)} className={selectClass}>
            <option value="">Any company</option>
            {companies.map((c) => (
              <option key={c.ticker} value={c.ticker}>
                {c.ticker} · {c.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={ids.yr} className={labelClass}>
            Fiscal year
          </label>
          <select id={ids.yr} value={year} onChange={(e) => setYear(e.target.value)} className={selectClass}>
            <option value="">Any year</option>
            {YEARS.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={ids.form} className={labelClass}>
            Form
          </label>
          <select id={ids.form} value={form} onChange={(e) => setForm(e.target.value)} className={selectClass}>
            <option value="">Any form</option>
            {FORMS.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>
        <div className="col-span-2 sm:col-span-1">
          <label htmlFor={ids.model} className={labelClass}>
            Model
          </label>
          <select
            id={ids.model}
            value={selectedModel}
            onChange={(e) => setModel(e.target.value)}
            className={selectClass}
            disabled={models.length === 0}
          >
            {models.length === 0 ? <option value="">Server default</option> : null}
            {models.map((m) => (
              <option key={m.id} value={m.id} title={`${m.id} via ${m.provider}`}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button type="submit" className={buttonClass.primary} disabled={busy || !question.trim()}>
          {busy ? <Spinner className="size-4" label="Answering" /> : null}
          {busy ? "Answering…" : "Ask"}
        </button>
        {busy ? (
          <button type="button" className={buttonClass.secondary} onClick={onCancel}>
            Stop
          </button>
        ) : null}
      </div>
      {!busy && !question ? (
        <div className="flex flex-wrap gap-2 pt-1" role="group" aria-label="Example questions">
          {EXAMPLE_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => setQuestion(q)}
              className="rounded-full border border-zinc-200 bg-white px-3 py-1 text-left text-xs text-zinc-600 hover:border-accent-300 hover:text-accent-800 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400 dark:hover:border-accent-700 dark:hover:text-accent-200"
            >
              {q}
            </button>
          ))}
        </div>
      ) : null}
    </form>
  );
}
