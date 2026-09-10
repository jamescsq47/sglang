"""Synthetic transport diagnostic, not a replacement dataset harness.

Preserves exact token history, simulates fast/slow tool waits, and records
request metadata and outputs. Production Retool/BrowseComp harnesses unchanged.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

import requests
from transformers import AutoTokenizer
from agentic_kv_request import (
    add_agentic_kv_metadata, build_agentic_extra_key,
    confirm_agentic_generation_tool, confirm_agentic_generation_final,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:29403")
    ap.add_argument("--model", default="/homes/siqic/Qwen3.5-9B")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--long-output", action="store_true", help="Cross Decode checkpoint boundaries")
    ap.add_argument("--archive-repeats", type=int, default=48)
    ap.add_argument("--one-token", action="store_true")
    ap.add_argument("--slow-first", action="store_true")
    args = ap.parse_args()
    os.environ["SGLANG_AGENTIC_KV_LIFECYCLE"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = []
    target = args.run_dir / f"multiturn-{uuid.uuid4().hex[:8]}.json"
    marker = ";" if args.one_token else "<END>"
    command = marker if args.one_token else ("echo cobalt=29; " * 12 if args.long_output else "echo cobalt=29") + marker
    for delay in ([3.0, 0.0] if args.slow_first else [0.0, 3.0]):
        metadata = {"agentic_request_id": f"q35-smoke-{uuid.uuid4().hex[:8]}"}
        text = "Archive: amber=17, cobalt=29, ivory=43. " * args.archive_repeats
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "Follow the requested output format exactly; no explanation."},
            {"role": "user", "content": text + "\nOutput exactly this tool call: " + command},
        ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(prompt, "keys"):
            prompt = prompt["input_ids"]
        if prompt and isinstance(prompt[0], list):
            prompt = prompt[0]
        prompt = [int(token) for token in prompt]
        for generation in range(3):
            sampling, request_id = add_agentic_kv_metadata(
                {"temperature": 0, "top_p": 1, "top_k": -1,
                 "max_new_tokens": 256, "stop": [marker], "no_stop_trim": True},
                trajectory_metadata=metadata, generation=generation,
                tokenizer=tokenizer, tool_type="transport_smoke",
                tool_suffix_markers=[marker], terminal_markers=[],
            )
            started = time.time()
            response = requests.post(args.url + "/generate", json={
                "input_ids": prompt, "sampling_params": sampling,
                "extra_key": build_agentic_extra_key(request_id, sampling),
                # /generate exposes exact output_ids without logprob. Native
                # P2D Host metadata intentionally does not carry logprobs.
                "return_logprob": False,
            }, timeout=180)
            response.raise_for_status()
            body = response.json()
            output = [int(token) for token in body["output_ids"]]
            records.append({"request_id": request_id, "generation": generation,
                            "tool_delay": delay, "seconds": time.time() - started,
                            "prompt_ids": list(prompt), "output_ids": output,
                            "response": body})
            target.write_text(json.dumps(records, ensure_ascii=False, indent=2))
            print(json.dumps({k: v for k, v in records[-1].items()
                              if k not in {"prompt_ids", "output_ids", "response"}}, ensure_ascii=False), flush=True)
            if body["meta_info"]["finish_reason"].get("type") in {"length", "abort"}:
                raise RuntimeError("diagnostic did not finish a reusable tool turn")
            if marker not in body.get("text", ""):
                confirm_agentic_generation_final(metadata, generation,
                    p_ready_dir=str(args.run_dir / "ready"))
                raise RuntimeError("diagnostic model ended without the requested tool marker")
            if generation == 2:
                confirm_agentic_generation_final(metadata, generation,
                    p_ready_dir=str(args.run_dir / "ready"))
            else:
                assert confirm_agentic_generation_tool(metadata, generation,
                    p_ready_dir=str(args.run_dir / "ready"))
                time.sleep(delay)
                suffix = tokenizer.encode(
                    "<|im_end|>\n<|im_start|>user\n"
                    + f"Tool result: command completed. Repeat exactly the same tool call as your previous response, ending with {marker}."
                    + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    add_special_tokens=False)
                prompt = list(prompt) + output + suffix
    # Same PD topology, same exact prompt, but no parent-generation reuse.
    # Run after the fast trajectories to avoid spending their Direct deadline
    # on an unrelated reference request.
    for record in records:
        metadata = {"agentic_request_id": f"q35-reference-{uuid.uuid4().hex[:8]}"}
        sampling, request_id = add_agentic_kv_metadata(
            {"temperature": 0, "top_p": 1, "top_k": -1,
             "max_new_tokens": 256, "stop": [marker], "no_stop_trim": True},
            trajectory_metadata=metadata, generation=0, tokenizer=tokenizer,
            tool_type="transport_reference", tool_suffix_markers=[], terminal_markers=[marker],
        )
        response = requests.post(args.url + "/generate", json={
            "input_ids": record["prompt_ids"], "sampling_params": sampling,
            "extra_key": build_agentic_extra_key(request_id, sampling),
            "return_logprob": False,
        }, timeout=180)
        response.raise_for_status()
        body = response.json()
        output = [int(token) for token in body["output_ids"]]
        record["recompute_reference"] = body
        record["exact_output_match"] = output == record["output_ids"]
        target.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"Reference generation={record['generation']} delay={record['tool_delay']} exact={record['exact_output_match']}", flush=True)
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
