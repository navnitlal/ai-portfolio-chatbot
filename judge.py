from __future__ import annotations
import json
from typing import Tuple, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from prompts import JUDGE_PROMPT

def _extract_json(text: str) -> Dict[str, Any] | None:
    """Try to parse JSON from judge output; model may return prose + JSON."""
    text = (text or "").strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first { and then matching } by brace count
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def judge_and_rewrite(judge_llm: BaseChatModel, main_llm: BaseChatModel, response_text: str, threshold: int = 7) -> Tuple[str, Dict[str, Any]]:
    judge_msg = judge_llm.invoke([HumanMessage(content=JUDGE_PROMPT + "\n\nRESPONSE:\n" + response_text)])
    verdict: Dict[str, Any] = {"score": 10, "issues": [], "rewrite_needed": False}
    raw = (judge_msg.content or "").strip()
    parsed = _extract_json(raw)
    if parsed:
        verdict = parsed
    else:
        verdict = {"score": threshold - 1, "issues": ["Judge output was not valid JSON."], "rewrite_needed": True}

    score = int(verdict.get("score", threshold - 1))
    rewrite_needed = bool(verdict.get("rewrite_needed", score < threshold))
    if not rewrite_needed:
        return response_text, verdict

    issues = verdict.get("issues", [])
    rewrite_prompt = f"""You are rewriting a chatbot reply that will be shown directly to the user. Output ONLY the improved reply text—nothing else.

Rules:
- Your entire output becomes the assistant's message. Do NOT include scores, ratings, "Quick fixes", "I can rewrite", "Score: X/10", or any meta-commentary.
- Fix the issues listed below in the original response. Keep the reply short and conversational (one short paragraph or a few bullets at most).
- Do NOT add command references, syntax lists, or manuals. Do NOT mention confirmations or permanent changes; this app does not perform permanent changes in chat.

Issues to fix: {issues}

Original response:
{response_text}
""".strip()
    improved = main_llm.invoke([HumanMessage(content=rewrite_prompt)]).content
    # If the rewriter still produced judge-like prose, keep the original response
    if improved and (
        "score" in improved.lower()
        or "quick fix" in improved.lower()
        or "i can rewrite" in improved.lower()
        or "i won't make any permanent" in improved.lower()
    ):
        return response_text, verdict
    return (improved or response_text), verdict
