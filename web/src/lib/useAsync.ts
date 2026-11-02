import { useEffect, useState } from "react";

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "error"; error: Error }
  | { status: "ready"; data: T };

/** Run an abortable loader whenever `key` changes. */
export function useAsync<T>(load: (signal: AbortSignal) => Promise<T>, key: string): AsyncState<T> {
  const [state, setState] = useState<{ key: string; value: AsyncState<T> }>({
    key,
    value: { status: "loading" },
  });

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setState({ key, value: { status: "ready", data } });
      },
      (error: unknown) => {
        if (controller.signal.aborted) return;
        const err = error instanceof Error ? error : new Error(String(error));
        setState({ key, value: { status: "error", error: err } });
      },
    );
    return () => controller.abort();
    // `load` is expected to be stable per key; re-running on identity would refetch every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  // A stale result for a previous key reads as loading.
  return state.key === key ? state.value : { status: "loading" };
}
