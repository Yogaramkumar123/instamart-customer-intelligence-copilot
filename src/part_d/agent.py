"""Autonomous ReAct refund agent with safety limits, ledger recording, and checkpointing."""
from datetime import datetime, timezone
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import mock_llm
from src.part_d.tool_registry import TOOL_MAP, TOOLS_SPEC
from src.part_d.checkpoint import save_checkpoint, load_checkpoint

AGENT_SYSTEM_PROMPT = (
    "You are an autonomous AGENT designed to triage and resolve customer refund requests. "
    "You have access to tools to query the order database, search policies, calculate refunds, "
    "issue approved refunds, and escalate high-value cases to a human lead. "
    "Execute the appropriate tools sequentially to reach a grounded resolution."
)


class SimulatedAgentCrash(Exception):
    """Simulated crash exception for verifying persistent checkpoint recovery."""


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    run_id: Optional[str] = None,
    crash_after_tool_n: Optional[int] = None,
    state_dir: str = "state",
    ledger_path: str = "outputs/refund_ledger.jsonl",
) -> Dict[str, Any]:
    """Execute the autonomous ReAct agent loop with safety limits and checkpointing.
    
    Args:
        user_message: Inbound customer message / refund request.
        scenario_id: Identifier for scenario (e.g., S1..S7).
        max_iterations: Maximum model invocations allowed (default 6).
        token_limit: Cumulative token ceiling (default 6000).
        time_limit_seconds: Wall-clock timeout limit in seconds (default 10).
        run_id: Unique run identifier (generated if None).
        crash_after_tool_n: Optional hook to simulate a crash immediately after tool n.
        state_dir: Directory to persist execution checkpoints.
        ledger_path: Path to outputs/refund_ledger.jsonl.
        
    Returns:
        Dictionary with status, final_answer, steps, llm_calls, tokens_used.
    """
    if run_id is None:
        run_id = f"run_{scenario_id}_{uuid.uuid4().hex[:8]}"

    start_time = time.monotonic()
    tokens_used = 0
    llm_calls = 0
    steps: List[Dict[str, Any]] = []
    tools_executed: List[str] = []

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        # Limit C: Wall-clock Timeout check at top of iteration
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

        # Limit A: Max iterations check before next call
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

        # Model invocation
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        # Limit B: Token ceiling check immediately after model call
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

        # Terminal text answer without tool calls
        if not msg.tool_calls:
            final_content = msg.content or ""
            # Normal status is the word before the colon
            if ":" in final_content:
                status = final_content.split(":")[0].strip()
            else:
                status = final_content.strip()

            refund_amt = 0.0
            for s in steps:
                if s["tool"] in ("issue_refund", "escalate_to_human"):
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

        # Tool calls requested by model
        tool_calls_payload = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_payload})

        for tc in msg.tool_calls:
            tool_name = tc.name
            args = tc.arguments or {}

            # Execute tool through TOOL_MAP
            if tool_name not in TOOL_MAP:
                obs_data = {"error": f"Tool '{tool_name}' not recognized."}
            else:
                tool_fn = TOOL_MAP[tool_name]
                # Provide ledger path to issue/escalate tools if applicable
                if tool_name in ("issue_refund", "escalate_to_human"):
                    obs_data = tool_fn(ledger_path=ledger_path, **args)
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

            # Save checkpoint after every observation
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

            # Simulated crash hook
            if crash_after_tool_n is not None and len(tools_executed) == crash_after_tool_n:
                raise SimulatedAgentCrash(
                    f"Simulated crash after executing tool {len(tools_executed)}: {tool_name} (run_id: {run_id})"
                )
