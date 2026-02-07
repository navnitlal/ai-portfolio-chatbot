"""All 8 graph node creation functions.

Nodes: planner, agent, tools, risk_questionnaire, market_analysis, judge, reflect, policy.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.state import (
    extract_tool_call_info,
    MAX_MESSAGES_IN_CONTEXT,
    interrupt,
)
from agent.tools import get_binding_tools, run_tool
from agent.judge import judge_and_rewrite
from agent.policy import ExecutionFlags, enforce_policy
from core.prompts import SYSTEM_PROMPT, NUM_RISK_QUESTIONS, RISK_QUESTIONS, RISK_QUESTION_OPTIONS
from core.risk import score_risk_answers
from observability import logger, node_timer


# ---------------------------------------------------------------------------
# Helpers (node-internal)
# ---------------------------------------------------------------------------

def _data_availability_context(config: RunnableConfig) -> str:
    """Context snippet for system prompt based on portfolio data availability."""
    cfg = config.get("configurable") or {}
    date_range = cfg.get("portfolio_date_range")
    if date_range is not None:
        min_d, max_d = date_range
        return (
            f"\n\n[Context: User has portfolio data from {min_d} to {max_d}. "
            "Call get_performance with start_date and end_date. "
            "Do NOT say you don't have data.]"
        )
    store = cfg.get("store")
    if not store:
        return ""
    try:
        df = store.load_all()
        if df.empty:
            return (
                "\n\n[Context: NO portfolio data loaded. "
                "Direct the user to upload a CSV via the sidebar. "
                "Do NOT call tools that require portfolio data.]"
            )
        min_d, max_d = df["Date"].min(), df["Date"].max()
        return (
            f"\n\n[Context: User has portfolio data from {min_d} to {max_d}. "
            "Call get_performance with start_date and end_date.]"
        )
    except Exception:
        return ""


def _is_likely_tool_output(content: str) -> bool:
    """True if content looks like a direct tool result; skip judge/policy to save latency."""
    if not (content or "").strip():
        return False
    c = content.strip()
    return (
        c.startswith("**Question ") and " of 4**" in c
        or c.startswith("Performance for ")
        or c.startswith("Your current risk category")
        or c.startswith("Based on your current portfolio allocation")
        or c.startswith("**As of ")
        or c.startswith("Latest trend summary")
        or c.startswith("Portfolio changes require")
        or c.startswith("Your inferred risk profile is:")
        or "I cannot change your portfolio" in c
        or "I cannot edit your portfolio" in c
        or "I don't have permission to edit your portfolio" in c
    )


# ---------------------------------------------------------------------------
# Planner Node (Chain-of-Thought)
# ---------------------------------------------------------------------------

def create_planner_node():
    """Chain-of-thought planner: classifies intent and outlines steps.
    Uses rule-based matching for common requests (zero latency),
    falls back to an LLM call for complex/ambiguous queries."""

    _SIMPLE_PLANS: Dict[str, str] = {
        "performance": "Call get_performance with the full date range from stored data.",
        "return": "Call get_performance with the full date range.",
        "drawdown": "Call get_performance to compute max drawdown.",
        "risk profile": "Call get_current_risk to infer risk from allocation.",
        "what's my risk": "Call get_current_risk.",
        "what is my risk": "Call get_current_risk.",
        "trend": "Call fetch_market_trends for relevant asset classes.",
        "market": "Call fetch_market_trends.",
        "outlook": "Call fetch_market_trends.",
        "questionnaire": "Call get_next_risk_question to start the questionnaire.",
        "risk question": "Call get_next_risk_question.",
        "rebalance": "Call suggest_rebalance with appropriate target_risk.",
        "what can you do": "Call get_capabilities.",
        "help": "Call get_capabilities.",
    }

    def planner_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        with node_timer("planner"):
            messages = state.get("messages", [])

            # Only plan when the last message is from the user
            if not messages or not isinstance(messages[-1], HumanMessage):
                return {"plan": "", "reflection_count": 0}

            last_human = (messages[-1].content or "").lower()

            # Fast rule-based plan for common requests
            for keyword, plan in _SIMPLE_PLANS.items():
                if keyword in last_human:
                    logger.info("Planner: rule match '%s'", keyword)
                    return {"plan": plan, "reflection_count": 0}

            # Complex request: use LLM
            cfg = config.get("configurable") or {}
            main_llm = cfg.get("main_llm")
            if not main_llm:
                return {"plan": "", "reflection_count": 0}

            plan_prompt = (
                "You are the planning step for a read-only portfolio assistant. "
                "Analyze the user's request and outline the action plan in 1-3 sentences.\n\n"
                "Available tools: get_performance (start_date, end_date, asset_class), "
                "get_current_risk, suggest_rebalance (target_risk), get_next_risk_question, "
                "submit_risk_answer, fetch_market_trends (asset_classes), get_capabilities, "
                "get_edit_help, apply_risk_profile_change.\n\n"
                f"User request: {messages[-1].content}"
            )

            try:
                response = main_llm.invoke([SystemMessage(content=plan_prompt)])
                plan = response.content or ""
                logger.info("Planner: LLM plan generated (%d chars)", len(plan))
                return {"plan": plan, "reflection_count": 0}
            except Exception as e:
                logger.warning("Planner LLM failed: %s", e)
                return {"plan": "", "reflection_count": 0}

    return planner_node


# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------

def create_agent_node(model_with_tools):
    """Agent node: calls the LLM with tools. Uses the plan from the planner node."""

    def agent_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        with node_timer("agent"):
            messages = list(state.get("messages", []))
            plan = state.get("plan", "")

            # Trim old messages to stay within context limits
            if len(messages) > MAX_MESSAGES_IN_CONTEXT:
                keep = messages[-MAX_MESSAGES_IN_CONTEXT:]
                while keep and isinstance(keep[0], ToolMessage):
                    keep = keep[1:]
                messages = keep

            # Build system prompt with plan and data context
            system_content = SYSTEM_PROMPT + _data_availability_context(config)
            if plan:
                system_content += f"\n\n[Plan for this request: {plan}]"

            # Prepend / refresh system prompt
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_content)] + messages
            else:
                messages = [SystemMessage(content=system_content)] + messages[1:]

            response = model_with_tools.invoke(messages)
            logger.info(
                "Agent: tool_calls=%s, content_len=%d",
                bool(getattr(response, "tool_calls", None)),
                len(response.content or ""),
            )
            return {"messages": [response]}

    return agent_node


# ---------------------------------------------------------------------------
# Tools Node (Parallel Execution + Error Recovery)
# ---------------------------------------------------------------------------

def create_tools_node():
    """Tools node: executes tool calls with parallel execution and per-tool error recovery."""

    # Tools that must run sequentially (order-dependent)
    _SEQUENTIAL_TOOLS = {"submit_risk_answer", "get_next_risk_question"}

    def tools_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        with node_timer("tools"):
            messages = list(state.get("messages", []))
            cfg = config.get("configurable") or {}

            last_msg = messages[-1] if messages else None
            if not isinstance(last_msg, AIMessage) or not getattr(last_msg, "tool_calls", None):
                return {}

            tool_calls = last_msg.tool_calls or []
            state_dict = {k: v for k, v in state.items() if k != "messages"}
            updates: Dict[str, Any] = {}

            # Separate parallel-safe and sequential tool calls
            parallel_tcs: list = []
            sequential_tcs: list = []
            for tc in tool_calls:
                name, args, tid = extract_tool_call_info(tc)
                if isinstance(args, dict) and "__tool_name__" in args:
                    args = {k: v for k, v in args.items() if k != "__tool_name__"}
                if name in _SEQUENTIAL_TOOLS:
                    sequential_tcs.append((name, args, tid))
                else:
                    parallel_tcs.append((name, args, tid))

            tool_messages: List[ToolMessage] = []

            # --- Parallel execution ---
            if len(parallel_tcs) > 1:
                logger.info("Tools: executing %d calls in parallel", len(parallel_tcs))
                with ThreadPoolExecutor(max_workers=min(len(parallel_tcs), 4)) as pool:
                    future_map = {}
                    for name, args, tid in parallel_tcs:
                        future = pool.submit(
                            run_tool, name, args, dict(state_dict), {"configurable": cfg}
                        )
                        future_map[future] = (name, tid)

                    for future in as_completed(future_map):
                        name, tid = future_map[future]
                        try:
                            result_text, state_updates = future.result()
                            updates.update(state_updates)
                        except Exception as e:
                            logger.error("Tool '%s' failed: %s", name, e)
                            result_text = f"[Tool error: {type(e).__name__}: {e}. Please try again.]"
                        tool_messages.append(
                            ToolMessage(content=result_text, tool_call_id=tid, name=name)
                        )
            elif parallel_tcs:
                name, args, tid = parallel_tcs[0]
                try:
                    result_text, state_updates = run_tool(name, args, state_dict, {"configurable": cfg})
                    updates.update(state_updates)
                except Exception as e:
                    logger.error("Tool '%s' failed: %s", name, e)
                    result_text = f"[Tool error: {type(e).__name__}: {e}. Please try again.]"
                tool_messages.append(
                    ToolMessage(content=result_text, tool_call_id=tid, name=name)
                )

            # --- Sequential execution ---
            state_dict.update(updates)
            for name, args, tid in sequential_tcs:
                risk_answers = state_dict.get("risk_answers") or {}
                if name == "submit_risk_answer" and not risk_answers:
                    tool_messages.append(ToolMessage(
                        content="[No answer recorded yet; show the first question from get_next_risk_question above.]",
                        tool_call_id=tid, name=name,
                    ))
                    continue
                try:
                    result_text, state_updates = run_tool(name, args, state_dict, {"configurable": cfg})
                    updates.update(state_updates)
                    state_dict.update(state_updates)
                except Exception as e:
                    logger.error("Tool '%s' failed: %s", name, e)
                    result_text = f"[Tool error: {type(e).__name__}: {e}. Please try again.]"
                tool_messages.append(
                    ToolMessage(content=result_text, tool_call_id=tid, name=name)
                )

            if not tool_messages:
                return updates
            return {"messages": tool_messages, **updates}

    return tools_node


# ---------------------------------------------------------------------------
# Risk Questionnaire Node (HITL with interrupt) + Subgraph
# ---------------------------------------------------------------------------

def create_risk_questionnaire_node():
    """Risk questionnaire node with Human-in-the-Loop via interrupt().
    When a question is generated, the graph pauses (interrupt) and waits
    for the user's answer. The app resumes with Command(resume=answer).
    Falls back to non-interrupt behavior if interrupt is unavailable."""

    def risk_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        with node_timer("risk_questionnaire"):
            messages = list(state.get("messages", []))
            cfg = config.get("configurable") or {}
            last_msg = messages[-1] if messages else None
            tool_calls = (
                getattr(last_msg, "tool_calls", []) or []
                if isinstance(last_msg, AIMessage) else []
            )
            state_dict = {k: v for k, v in state.items() if k != "messages"}

            tool_messages: List[ToolMessage] = []
            updates: Dict[str, Any] = {}

            for tc in tool_calls:
                name, args, tid = extract_tool_call_info(tc)
                if name not in ("get_next_risk_question", "submit_risk_answer"):
                    continue

                try:
                    result_text, state_updates = run_tool(
                        name, args, state_dict, {"configurable": cfg}
                    )
                    updates.update(state_updates)
                    state_dict.update(state_updates)
                except Exception as e:
                    logger.error("Risk tool '%s' failed: %s", name, e)
                    tool_messages.append(
                        ToolMessage(content=f"[Error: {e}]", tool_call_id=tid, name=name)
                    )
                    continue

                # HITL: if the result is a question, interrupt for user input
                is_question = "**Question " in result_text and " of 4**" in result_text
                if interrupt is not None and is_question:
                    logger.info("Risk: interrupting for user input")

                    # Parse question metadata for the app
                    q_match = re.search(r"\*\*Question (\d) of", result_text)
                    q_num = int(q_match.group(1)) if q_match else 1
                    options: list = []
                    if "Choose one:" in result_text:
                        rest = result_text.split("Choose one:")[1].split("\n")[0].strip()
                        options = [s.strip().strip("*").strip() for s in rest.split("|") if s.strip()]

                    # Pause graph — app displays question and collects answer
                    user_answer = interrupt({
                        "type": "risk_question",
                        "question_number": q_num,
                        "question_text": result_text,
                        "options": options,
                    })

                    logger.info("Risk: resumed with answer '%s' for Q%d", user_answer, q_num)

                    # Process the answer
                    submit_result, submit_updates = run_tool(
                        "submit_risk_answer",
                        {"question_number": str(q_num), "answer_text": user_answer},
                        state_dict,
                        {"configurable": cfg},
                    )
                    updates.update(submit_updates)
                    state_dict.update(submit_updates)
                    tool_messages.append(
                        ToolMessage(content=submit_result, tool_call_id=tid, name=name)
                    )
                else:
                    # No interrupt available or result is the final profile — pass through
                    tool_messages.append(
                        ToolMessage(content=result_text, tool_call_id=tid, name=name)
                    )

            if not tool_messages:
                return updates
            return {"messages": tool_messages, **updates}

    return risk_node


# ---------------------------------------------------------------------------
# Market Analysis Node (Subgraph with Error Recovery)
# ---------------------------------------------------------------------------

def create_market_analysis_node():
    """Market analysis node: fetches trends via Tavily with retry-based error recovery.
    Encapsulates the search + summarize pipeline as a conceptual subgraph with
    per-step error handling."""

    def market_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        with node_timer("market_analysis"):
            messages = list(state.get("messages", []))
            cfg = config.get("configurable") or {}
            last_msg = messages[-1] if messages else None
            tool_calls = (
                getattr(last_msg, "tool_calls", []) or []
                if isinstance(last_msg, AIMessage) else []
            )
            state_dict = {k: v for k, v in state.items() if k != "messages"}

            tool_messages: List[ToolMessage] = []
            updates: Dict[str, Any] = {}

            for tc in tool_calls:
                name, args, tid = extract_tool_call_info(tc)
                if name != "fetch_market_trends":
                    continue

                # Error recovery: retry up to 2 times
                max_attempts = 2
                for attempt in range(1, max_attempts + 1):
                    try:
                        result_text, state_updates = run_tool(
                            name, args, state_dict, {"configurable": cfg}
                        )
                        updates.update(state_updates)
                        logger.info("Market analysis: succeeded on attempt %d", attempt)
                        break
                    except Exception as e:
                        logger.warning("Market analysis attempt %d/%d failed: %s", attempt, max_attempts, e)
                        if attempt == max_attempts:
                            result_text = (
                                "Market trends are temporarily unavailable due to a service issue. "
                                "Please try again in a moment."
                            )

                tool_messages.append(
                    ToolMessage(content=result_text, tool_call_id=tid, name=name)
                )

            if not tool_messages:
                return updates
            return {"messages": tool_messages, **updates}

    return market_node


# ---------------------------------------------------------------------------
# Judge Node
# ---------------------------------------------------------------------------

def create_judge_node():
    """Judge node: scores the response and optionally rewrites it."""

    def judge_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        with node_timer("judge"):
            messages = list(state.get("messages", []))
            cfg = config.get("configurable") or {}
            judge_llm = cfg.get("judge_llm")
            main_llm = cfg.get("main_llm")

            skip_result = {"last_judge": {"score": 10, "issues": [], "rewrite_needed": False}}

            if not judge_llm or not main_llm:
                return skip_result
            last_msg = messages[-1] if messages else None
            if not isinstance(last_msg, AIMessage):
                return skip_result
            last_content = last_msg.content or ""
            if _is_likely_tool_output(last_content):
                return skip_result

            improved, verdict = judge_and_rewrite(judge_llm, main_llm, last_content)
            updates: Dict[str, Any] = {"last_judge": verdict}

            if improved != last_content:
                updated_messages = messages[:-1] + [AIMessage(content=improved)]
                updates["messages"] = updated_messages

            return updates

    return judge_node


# ---------------------------------------------------------------------------
# Reflect Node (Self-Correction)
# ---------------------------------------------------------------------------

def create_reflect_node():
    """Passes judge feedback back to the agent for self-correction.
    Increments reflection_count so the loop runs at most MAX_REFLECTIONS times."""

    def reflect_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        with node_timer("reflect"):
            last_judge = state.get("last_judge", {})
            issues = last_judge.get("issues", [])
            messages = list(state.get("messages", []))

            # Get original response for context
            original = ""
            if messages and isinstance(messages[-1], AIMessage):
                original = (messages[-1].content or "")[:300]

            feedback = SystemMessage(
                content=(
                    "[Reflection required] The quality judge found issues:\n"
                    + "\n".join(f"- {issue}" for issue in issues)
                    + f"\n\nOriginal response: {original}...\n\n"
                    "Produce a revised response addressing these issues. "
                    "Keep it concise and conversational. Output ONLY the revised text."
                )
            )

            logger.info("Reflect: sending %d issues back to agent", len(issues))
            return {
                "messages": [feedback],
                "reflection_count": state.get("reflection_count", 0) + 1,
            }

    return reflect_node


# ---------------------------------------------------------------------------
# Policy Node
# ---------------------------------------------------------------------------

def create_policy_node():
    """Policy guardrail: ensures the assistant doesn't claim unperformed actions."""

    def policy_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        with node_timer("policy"):
            messages = list(state.get("messages", []))
            cfg = config.get("configurable") or {}
            policy_llm = cfg.get("policy_llm")
            store = cfg.get("store")

            if not policy_llm:
                return {"last_policy": {"ok": True, "issues": []}}
            last_msg = messages[-1] if messages else None
            if not isinstance(last_msg, AIMessage):
                return {"last_policy": {"ok": True, "issues": []}}
            last_content = last_msg.content or ""
            if _is_likely_tool_output(last_content):
                return {"last_policy": {"ok": True, "issues": []}}

            last_user_text = ""
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    last_user_text = (m.content or "").lower()
                    break

            has_portfolio_data = False
            if store:
                try:
                    df = store.load_all()
                    has_portfolio_data = not df.empty
                except Exception:
                    pass

            flags = ExecutionFlags(
                did_persist_change=state.get("did_persist_change", False),
                did_use_tavily=state.get("did_use_tavily", False),
                did_compute_metrics=state.get("did_compute_metrics", False),
                has_portfolio_data=has_portfolio_data,
                has_data_in_requested_range=True,
                user_consented_tavily=False,
                user_requested_metrics=any(k in last_user_text for k in ["performance", "return", "drawdown"]),
                user_requested_trends=any(k in last_user_text for k in ["trend", "outlook", "market"]),
            )

            safe_text, policy_report = enforce_policy(policy_llm, last_content, flags)
            updates: Dict[str, Any] = {"last_policy": policy_report}

            if safe_text != last_content:
                updated_messages = messages[:-1] + [AIMessage(content=safe_text)]
                updates["messages"] = updated_messages

            # Reset execution flags for next turn
            updates["did_persist_change"] = False
            updates["did_use_tavily"] = False
            updates["did_compute_metrics"] = False

            return updates

    return policy_node
