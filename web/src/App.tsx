import { IS_STATIC, REPO_URL } from "./lib/env";
import { href, parseHash, useHash, type Route } from "./lib/router";
import { ChatPage } from "./pages/ChatPage";
import { EvalPage } from "./pages/EvalPage";
import { RunPage } from "./pages/RunPage";
import { cx } from "./lib/style";

function Logo() {
  return (
    <a href={IS_STATIC ? href.eval() : href.chat()} className="flex items-center gap-2 rounded-md" aria-label="LedgerLens home">
      <svg viewBox="0 0 32 32" className="size-6" aria-hidden="true">
        <rect width="32" height="32" rx="7" className="fill-accent-700 dark:fill-accent-500" />
        <path d="M10 8v16h12" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="20" cy="13" r="4" fill="none" stroke="white" strokeWidth="2.5" />
      </svg>
      <span className="text-[0.95rem] font-semibold tracking-tight">
        Ledger<span className="text-accent-700 dark:text-accent-400">Lens</span>
      </span>
    </a>
  );
}

function NavLink({ to, active, children }: { to: string; active: boolean; children: string }) {
  return (
    <a
      href={to}
      aria-current={active ? "page" : undefined}
      className={cx(
        "rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors",
        active
          ? "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
          : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100",
      )}
    >
      {children}
    </a>
  );
}

function Page({ route }: { route: Route }) {
  switch (route.name) {
    case "chat":
      return <ChatPage />;
    case "eval":
      return <EvalPage />;
    case "run":
      return <RunPage key={route.runId} runId={route.runId} qid={route.qid} />;
    case "notFound":
      return (
        <div className="py-16 text-center">
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Nothing at {route.path}.</p>
          <a href={href.eval()} className="mt-2 inline-block text-sm font-medium text-accent-700 hover:underline dark:text-accent-300">
            Go to the evaluation lab
          </a>
        </div>
      );
  }
}

export default function App() {
  const route = parseHash(useHash(), IS_STATIC ? "eval" : "chat");
  const evalActive = route.name === "eval" || route.name === "run";
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main")?.focus();
        }}
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-white focus:px-3 focus:py-2 focus:text-sm focus:shadow dark:focus:bg-zinc-900"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-40 border-b border-zinc-200 bg-white/85 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/85">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex items-center gap-4 sm:gap-6">
            <Logo />
            <nav aria-label="Main" className="flex items-center gap-1">
              <NavLink to={href.chat()} active={route.name === "chat"}>
                Ask
              </NavLink>
              <NavLink to={href.eval()} active={evalActive}>
                Eval lab
              </NavLink>
            </nav>
          </div>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="text-sm text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
          >
            GitHub
          </a>
        </div>
      </header>
      <main id="main" tabIndex={-1} className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 outline-none sm:px-6 sm:py-8">
        <Page route={route} />
      </main>
      <footer className="border-t border-zinc-200 dark:border-zinc-800">
        <div className="mx-auto max-w-7xl px-4 py-4 text-xs text-zinc-500 sm:px-6 dark:text-zinc-400">
          LedgerLens answers from public SEC filings. Not investment advice.
        </div>
      </footer>
    </div>
  );
}
