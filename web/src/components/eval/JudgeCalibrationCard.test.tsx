import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { JudgeCalibration } from "../../api/types";
import { JudgeCalibrationCard } from "./JudgeCalibrationCard";

const mv = (value: number) => ({ value, ci_low: value - 0.05, ci_high: value + 0.05, n: 150 });

const base: JudgeCalibration = {
  judge_model: "judge-model",
  prompt_version: "judge-v3",
  split: "test",
  n: 150,
  accuracy: mv(0.887),
  kappa: mv(0.802),
  error_rate: mv(0.113),
  binary_accuracy: mv(0.913),
  binary_kappa: mv(0.818),
  labels: ["CORRECT", "INCORRECT", "REFUSAL"],
  confusion: [
    [78, 5, 1],
    [6, 38, 2],
    [0, 3, 17],
  ],
  per_label_recall: { CORRECT: 0.9286, INCORRECT: 0.8261, REFUSAL: 0.85 },
};

describe("JudgeCalibrationCard", () => {
  it("renders the labeled matrix and calls the test split held-out", () => {
    render(<JudgeCalibrationCard calibration={base} />);
    expect(screen.getAllByRole("columnheader", { name: /Refusal/ }).length).toBeGreaterThan(0);
    expect(screen.getByText("Judge error rate")).toBeInTheDocument();
    expect(screen.getByText(/human labels \(held-out split\)\./)).toBeInTheDocument();
    expect(screen.queryByText(/Untouched by prompt tuning/)).toBeNull();
  });

  it("names other splits instead of calling them held-out", () => {
    render(<JudgeCalibrationCard calibration={{ ...base, split: "dev" }} />);
    expect(screen.getByText(/human labels \(split: dev\)\./)).toBeInTheDocument();
    expect(screen.queryByText(/held-out/)).toBeNull();
  });

  it("shows the subset untouched by prompt tuning when present", () => {
    render(
      <JudgeCalibrationCard
        calibration={{ ...base, clean_n: 60, clean_accuracy: mv(0.867), clean_kappa: mv(0.771) }}
      />,
    );
    expect(screen.getByText(/Untouched by prompt tuning \(n=60\)/)).toBeInTheDocument();
    expect(screen.getByText("86.7%")).toBeInTheDocument();
  });
});
