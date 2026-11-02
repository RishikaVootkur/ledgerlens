import type { Element, ElementContent, Root, RootContent, Text } from "hast";
import { quotePattern } from "./citations";

/**
 * Rehype plugin that wraps the cited quote in <mark> inside rendered Markdown.
 * Works per text node, so a quote that spans formatting or table cells is
 * simply left unhighlighted (the documented fallback).
 */
export function rehypeHighlightQuote(options: { quote: string }) {
  const pattern = quotePattern(options.quote);
  return (tree: Root) => {
    if (!pattern) return;
    visit(tree, pattern);
  };
}

type Parent = Root | Element;

function visit(node: Parent, pattern: RegExp): void {
  const next: Array<RootContent | ElementContent> = [];
  let changed = false;
  for (const child of node.children) {
    if (child.type === "text") {
      const parts = splitText(child, pattern);
      if (parts) {
        next.push(...parts);
        changed = true;
        continue;
      }
    } else if (child.type === "element" && child.tagName !== "mark") {
      visit(child, pattern);
    }
    next.push(child);
  }
  if (changed) node.children = next as typeof node.children;
}

function splitText(node: Text, pattern: RegExp): ElementContent[] | null {
  const match = pattern.exec(node.value);
  if (!match) return null;
  const before = node.value.slice(0, match.index);
  const after = node.value.slice(match.index + match[0].length);
  const mark: Element = {
    type: "element",
    tagName: "mark",
    properties: { className: ["quote-mark"], dataQuote: "true" },
    children: [{ type: "text", value: match[0] }],
  };
  const out: ElementContent[] = [];
  if (before) out.push({ type: "text", value: before });
  out.push(mark);
  if (after) out.push({ type: "text", value: after });
  return out;
}
