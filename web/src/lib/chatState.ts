import type { AgentAnswer, ChatEvent } from "../api/types";

export interface StepState {
  name: string;
  detail: string;
  status: "active" | "done";
}

export interface ChatState {
  status: "idle" | "streaming" | "done" | "error" | "cancelled";
  question: string;
  steps: StepState[];
  /** Answer text as streamed; replaced by the final answer when it arrives. */
  text: string;
  answer: AgentAnswer | null;
  error: string | null;
}

export type ChatAction =
  | { type: "start"; question: string }
  | { type: "event"; event: ChatEvent }
  | { type: "fail"; message: string }
  | { type: "cancel" };

export const initialChatState: ChatState = {
  status: "idle",
  question: "",
  steps: [],
  text: "",
  answer: null,
  error: null,
};

const finishSteps = (steps: StepState[]): StepState[] =>
  steps.map((s) => (s.status === "done" ? s : { ...s, status: "done" }));

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "start":
      return { ...initialChatState, status: "streaming", question: action.question };
    case "fail":
      return { ...state, status: "error", error: action.message, steps: finishSteps(state.steps) };
    case "cancel":
      return state.status === "streaming"
        ? { ...state, status: "cancelled", steps: finishSteps(state.steps) }
        : state;
    case "event":
      return applyEvent(state, action.event);
  }
}

function applyEvent(state: ChatState, event: ChatEvent): ChatState {
  if (state.status !== "streaming") return state;
  switch (event.type) {
    case "step":
      return {
        ...state,
        steps: [...finishSteps(state.steps), { ...event.data, status: "active" }],
      };
    case "token":
      return { ...state, text: state.text + event.data.text };
    case "answer":
      return { ...state, answer: event.data, text: event.data.answer };
    case "error":
      return { ...state, status: "error", error: event.data.message, steps: finishSteps(state.steps) };
    case "done":
      return { ...state, status: "done", steps: finishSteps(state.steps) };
  }
}
