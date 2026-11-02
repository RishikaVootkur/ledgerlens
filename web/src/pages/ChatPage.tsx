import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { getCompanies, getModels, streamChat } from "../api/client";
import type { ChatRequest, CompaniesResponse, ModelsResponse } from "../api/types";
import { ChatForm } from "../components/chat/ChatForm";
import { CitationsPanel } from "../components/chat/CitationsPanel";
import { StepTimeline } from "../components/chat/StepTimeline";
import { UsageBar } from "../components/chat/UsageBar";
import { Drawer } from "../components/Drawer";
import { AnswerMarkdown } from "../components/Markdown";
import { Badge, Card, ErrorNote } from "../components/ui";
import { buttonClass } from "../lib/style";
import { chatReducer, initialChatState } from "../lib/chatState";
import { IS_STATIC, MOCK_CHAT, RUN_LOCALLY_URL } from "../lib/env";
import { hashQuery, href } from "../lib/router";

function useOptional<T>(load: () => Promise<T>): T | null {
  const [value, setValue] = useState<T | null>(null);
  useEffect(() => {
    let alive = true;
    load().then(
      (v) => alive && setValue(v),
      () => undefined, // selects fall back to "any" / server default
    );
    return () => {
      alive = false;
    };
  }, [load]);
  return value;
}

const loadCompanies = (): Promise<CompaniesResponse> => getCompanies();
const loadModels = (): Promise<ModelsResponse> => getModels();

export function ChatPage() {
  if (IS_STATIC) return <StaticChatNotice />;
  return <LiveChat />;
}

function StaticChatNotice() {
  return (
    <div className="mx-auto max-w-2xl py-10">
      <Card className="p-6">
        <h1 className="text-lg font-semibold tracking-tight">Ask runs locally</h1>
        <p className="mt-2 text-sm leading-relaxed text-zinc-600 dark:text-zinc-400">
          This static site has no backend, so live questions are turned off. Clone the repo and run
          the API to ask questions with cited answers from SEC filings.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <a className={buttonClass.primary} href={RUN_LOCALLY_URL} target="_blank" rel="noreferrer">
            How to run it locally
          </a>
          <a className={buttonClass.secondary} href={href.eval()}>
            Open the eval lab
          </a>
        </div>
      </Card>
    </div>
  );
}

function LiveChat() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  const companies = useOptional(loadCompanies);
  const models = useOptional(loadModels);
  const companyByTicker = useMemo(
    () => new Map((companies?.companies ?? []).map((c) => [c.ticker, c.name])),
    [companies],
  );
  const companyName = useCallback((t: string) => companyByTicker.get(t), [companyByTicker]);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const ask = useCallback(async (request: ChatRequest) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setActiveIndex(null);
    dispatch({ type: "start", question: request.question });
    try {
      await streamChat(request, (event) => dispatch({ type: "event", event }), controller.signal);
      dispatch({ type: "event", event: { type: "done", data: {} } });
    } catch (err) {
      if (controller.signal.aborted) return;
      dispatch({ type: "fail", message: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  // #/ask?q=... prefills the question; the recorded demo also runs it, which
  // makes README screenshots reproducible. A live backend never auto-runs.
  const [initialQuestion] = useState(() => hashQuery().get("q") ?? "");
  useEffect(() => {
    if (!MOCK_CHAT || !initialQuestion) return;
    const timer = window.setTimeout(() => {
      void ask({ question: initialQuestion }).then(() => setActiveIndex((i) => i ?? 1));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [ask, initialQuestion]);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    dispatch({ type: "cancel" });
  }, []);

  const citations = state.answer?.citations;
  const selectCitation = useCallback((index: number) => {
    setActiveIndex(index);
    // The side panel is hidden below lg, so open the drawer there.
    if (!window.matchMedia("(min-width: 1024px)").matches) setDrawerOpen(true);
  }, []);
  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  const streaming = state.status === "streaming";
  const started = state.status !== "idle";

  const panel = (
    <CitationsPanel
      citations={citations ?? []}
      activeIndex={activeIndex}
      onSelect={setActiveIndex}
      companyName={companyName}
      pending={streaming}
    />
  );

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_400px]">
      <div className="min-w-0 space-y-5">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Ask the filings</h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Answers come from SEC 10-K and 10-Q filings, with a source behind every claim.
            {MOCK_CHAT ? " Demo mode: responses are replayed from a recording." : null}
          </p>
        </div>
        <Card className="p-4 sm:p-5">
          <ChatForm
            companies={companies?.companies ?? []}
            models={models?.models ?? []}
            defaultModel={models?.default ?? null}
            busy={streaming}
            initialQuestion={initialQuestion}
            onSubmit={ask}
            onCancel={cancel}
          />
        </Card>

        {started ? (
          <Card className="overflow-hidden" labelledBy="answer-heading">
            <div className="border-b border-zinc-200 px-4 py-3 sm:px-5 dark:border-zinc-800">
              <h2 id="answer-heading" className="sr-only">
                Answer
              </h2>
              <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">{state.question}</p>
            </div>
            <div className="grid gap-5 p-4 sm:p-5 md:grid-cols-[200px_minmax(0,1fr)]">
              <div className="md:border-r md:border-zinc-200 md:pr-4 md:dark:border-zinc-800">
                <h3 className="mb-2 text-xs font-medium tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
                  Steps
                </h3>
                {state.steps.length > 0 ? (
                  <StepTimeline steps={state.steps} live={streaming} />
                ) : (
                  <p className="text-sm text-zinc-500 dark:text-zinc-400">Starting…</p>
                )}
              </div>
              <div className="min-w-0 space-y-4" aria-live="polite" aria-busy={streaming}>
                {state.answer?.refused ? (
                  <Badge tone="warn">Refused: insufficient support</Badge>
                ) : null}
                {state.text ? (
                  <AnswerMarkdown
                    text={state.text}
                    citations={citations}
                    activeIndex={activeIndex}
                    onSelectCitation={selectCitation}
                    streaming={streaming}
                  />
                ) : streaming ? (
                  <p className="text-sm text-zinc-500 dark:text-zinc-400">Researching…</p>
                ) : null}
                {state.status === "error" && state.error ? (
                  <ErrorNote title="The answer could not be completed" error={state.error} />
                ) : null}
                {state.status === "cancelled" ? (
                  <p className="text-sm text-zinc-500 dark:text-zinc-400">Stopped.</p>
                ) : null}
                {citations && citations.length > 0 ? (
                  <button
                    type="button"
                    className={`${buttonClass.secondary} lg:hidden`}
                    onClick={() => {
                      setActiveIndex((i) => i ?? 1);
                      setDrawerOpen(true);
                    }}
                  >
                    View sources ({citations.length})
                  </button>
                ) : null}
                {state.answer ? (
                  <div className="border-t border-zinc-200 pt-3 dark:border-zinc-800">
                    <UsageBar usage={state.answer.usage} />
                  </div>
                ) : null}
              </div>
            </div>
          </Card>
        ) : null}
      </div>

      <aside className="hidden lg:block" aria-labelledby="sources-heading">
        <div className="sticky top-20">
          <Card className="max-h-[calc(100vh-6rem)] overflow-y-auto p-4">
            <h2 id="sources-heading" className="mb-3 text-sm font-semibold">
              Sources{citations && citations.length > 0 ? ` (${citations.length})` : ""}
            </h2>
            {panel}
          </Card>
        </div>
      </aside>

      <Drawer open={drawerOpen} onClose={closeDrawer} title="Sources">
        {panel}
      </Drawer>
    </div>
  );
}
