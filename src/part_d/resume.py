"""Resume agent execution from persistent disk checkpoint."""
from datetime import datetime, timezone
import json
import time
from typing import Any, Dict, List, Optional

import mock_llm
from src.part_d.checkpoint import load_checkpoint, save_checkpoint
from src.part_d.tool_registry import TOOL_MAP, TOOLS_SPEC


def resume_agent(
    run_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    state_dir: str = "state",
    ledger_path: str = "outputs/refund_ledger.jsonl",
) -> Dict[str, Any]:
    """Resume execution of an agent from a persistent checkpoint on disk.
    
    Restores the exact conversation history, tokens used, call counts,
    and step numbers, and continues the ReAct loop without re-executing
    previously completed tools.
    """
    checkpoint = load_checkpoint(run_id, state_dir=state_dir)

    scenario_id = checkpoint["scenario_id"]
    messages = list(checkpoint["messages"])
    tokens_used = int(checkpoint["tokens_used"])
    llm_calls = int(checkpoint["llm_calls"])
    steps = list(checkpoint["steps"])
    tools_executed = list(checkpoint["tools_executed"])

    start_time = time.monotonic()

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: Wall-clock timeout limit reached; handing off to human support.",
                "refund_amount": 0.0,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        if llm_calls >= max_iterations:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_MAX_ITERATIONS",
                "final_answer": "FALLBACK_MAX_ITERATIONS: Maximum iterations reached without final resolution; handing off to human support.",
                "refund_amount": 0.0,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        if tokens_used > token_limit:
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: Token consumption ceiling exceeded; handing off to human support.",
                "refund_amount": 0.0,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            final_content = msg.content or ""
            if ":" in final_content:
                status = final_content.split(":")[0].strip()
            else:
                status = final_content.strip()

            refund_amt = 0.0
            for s in steps:
                if s["tool"] in ("issue_refund", "escalate_to_human"):
                    obs = s.get("observation", {})
                    if "error" not in obs:
                        refund_amt = float(s["arguments"].get("refund_inr", 0.0))

            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_content,
                "refund_amount": refund_amt,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "tools_executed": tools_executed,
            }

        tool_calls_payload = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_payload})

        for tc in msg.tool_calls:
            tool_name = tc.name
            args = dict(tc.arguments or {})

            if tool_name not in TOOL_MAP:
                obs_data = {"error": f"Tool '{tool_name}' not recognized."}
            else:
                tool_fn = TOOL_MAP[tool_name]
                if tool_name in ("issue_refund", "escalate_to_human"):
                    call_kwargs = dict(args)
                    if "idempotency_key" not in call_kwargs or not call_kwargs["idempotency_key"]:
                        call_kwargs["idempotency_key"] = f"{run_id}:{call_kwargs.get('order_id', 'unknown')}"
                    obs_data = tool_fn(ledger_path=ledger_path, **call_kwargs)
                else:
                    obs_data = tool_fn(**args)

            obs_content = json.dumps(obs_data)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": obs_content,
            })

            tools_executed.append(tool_name)
            steps.append({
                "step": len(steps) + 1,
                "tool": tool_name,
                "arguments": args,
                "observation": obs_data,
            })

            checkpoint_data = {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "messages": messages,
                "tokens_used": tokens_used,
                "llm_calls": llm_calls,
                "step_count": len(steps),
                "tools_executed": list(tools_executed),
                "steps": steps,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_checkpoint(run_id, checkpoint_data, state_dir=state_dir)
