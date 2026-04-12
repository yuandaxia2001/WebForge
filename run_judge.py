#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_judge.py — Judge agent answers against ground truth using an LLM.

Usage:
    # Judge all completed tasks
    python run_judge.py --config config.yaml --output-dir ./output

    # Judge and print summary by domain/level (requires tasks.jsonl for metadata)
    python run_judge.py --config config.yaml --output-dir ./output --task-file tasks.jsonl
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

import openai
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("webforge-judge")


# ─────────────── Judge Prompt ───────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating whether an AI agent correctly completed a web-based task.

You will be given:
1. The original task prompt (what the agent was asked to do).
2. The ground truth answer (the expected correct answer).
3. The agent's answer (what the agent actually produced).

Your job is to determine if the agent's answer is **correct** by comparing it to the ground truth.

Evaluation rules:
- Focus on semantic equivalence, not exact string matching.
- Minor formatting differences (e.g., extra spaces, capitalization, currency symbols) should be ignored.
- If the ground truth is a specific code/reference number, the agent must produce that exact code.
- If the ground truth is a descriptive answer, the agent's answer should convey the same meaning.
- If the agent's answer contains the ground truth information among other text, it is still correct.
- If the agent failed the task or produced an irrelevant answer, mark it as incorrect.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{"correct": true, "reason": "brief explanation"}
or
{"correct": false, "reason": "brief explanation"}
"""

JUDGE_USER_PROMPT_TEMPLATE = """\
## Task Prompt
{task_prompt}

## Ground Truth Answer
{ground_truth}

## Agent's Answer
{agent_answer}

Please judge whether the agent's answer is correct."""


def load_config(config_path: str) -> dict:
    """Load YAML config and extract judge_llm settings."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    judge_cfg = raw.get("judge_llm", {})
    if not judge_cfg.get("model"):
        LOGGER.error("Config error: judge_llm.model is required")
        sys.exit(1)
    if not judge_cfg.get("api_key") or judge_cfg["api_key"] == "your-judge-api-key-here":
        LOGGER.error("Config error: judge_llm.api_key must be set")
        sys.exit(1)

    return judge_cfg


async def judge_answer(
    client: openai.AsyncOpenAI,
    model: str,
    task_prompt: str,
    ground_truth: str,
    agent_answer: str,
    max_retries: int = 3,
) -> tuple:
    """Use the judge LLM to evaluate if the agent's answer is correct."""
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        ground_truth=ground_truth,
        agent_answer=agent_answer,
    )

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("correct", False), result.get("reason", "")
            else:
                LOGGER.warning(f"Judge returned non-JSON: {content}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return False, f"Failed to parse judge response: {content}"
        except Exception as e:
            LOGGER.error(f"Judge LLM error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
            else:
                return False, f"Judge error: {e}"

    return False, "Max retries exceeded"


def collect_results(output_dir: Path) -> list:
    """Collect all result.json files from the output directory."""
    results = []
    for task_dir in sorted(output_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        result_file = task_dir / "result.json"
        if not result_file.exists():
            continue
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["task_id"] = task_dir.name
        results.append(data)
    return results


def load_task_metadata(task_file: str) -> dict:
    """Load task metadata (domain, level) from tasks.jsonl."""
    metadata = {}
    if not task_file or not os.path.exists(task_file):
        return metadata
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            metadata[task["id"]] = task
    return metadata


async def main():
    parser = argparse.ArgumentParser(
        description="WebForge-Bench Judge — Evaluate agent answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_judge.py --config config.yaml --output-dir ./output
  python run_judge.py --config config.yaml --output-dir ./output --task-file tasks.jsonl
""",
    )
    parser.add_argument("--config", required=True, help="YAML config file (must contain judge_llm section)")
    parser.add_argument("--output-dir", required=True, help="Directory containing task results (output/{task_id}/result.json)")
    parser.add_argument("--task-file", default=None, help="Path to tasks.jsonl (optional, for domain/level breakdown)")
    parser.add_argument("--save-csv", default=None, help="Save per-task results to CSV file")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent judge requests (default: 10)")
    args = parser.parse_args()

    # Load config
    judge_cfg = load_config(args.config)
    client = openai.AsyncOpenAI(
        api_key=judge_cfg["api_key"],
        base_url=judge_cfg.get("base_url", ""),
    )
    model = judge_cfg["model"]

    # Collect results
    output_dir = Path(args.output_dir)
    results = collect_results(output_dir)
    LOGGER.info(f"Found {len(results)} completed task(s) in {output_dir}")

    if not results:
        LOGGER.error("No results found. Run evaluation first with run_eval.py.")
        sys.exit(1)

    # Load task metadata if available
    task_metadata = load_task_metadata(args.task_file) if args.task_file else {}

    # Judge all tasks with concurrency control
    semaphore = asyncio.Semaphore(args.max_concurrent)
    judge_results = []

    async def judge_one(result):
        async with semaphore:
            task_id = result["task_id"]
            answer = result.get("answer", "")
            ground_truth = result.get("ground_truth", "")
            task_prompt = result.get("task_prompt", "")

            # Skip tasks where agent explicitly failed
            if not result.get("success", False) and not answer:
                return {
                    "task_id": task_id,
                    "correct": False,
                    "reason": "Agent reported failure with no answer",
                    "answer": answer,
                    "ground_truth": ground_truth,
                }

            correct, reason = await judge_answer(client, model, task_prompt, ground_truth, answer)
            LOGGER.info(f"[{task_id}] {'✓' if correct else '✗'} {reason[:80]}")

            return {
                "task_id": task_id,
                "correct": correct,
                "reason": reason,
                "answer": answer,
                "ground_truth": ground_truth,
            }

    tasks = [judge_one(r) for r in results]
    judge_results = await asyncio.gather(*tasks)

    # Save per-task judge results back to result.json
    for jr in judge_results:
        result_file = output_dir / jr["task_id"] / "result.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["judge_correct"] = jr["correct"]
            data["judge_reason"] = jr["reason"]
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # ─────────────── Summary ───────────────
    total = len(judge_results)
    correct = sum(1 for r in judge_results if r["correct"])
    accuracy = correct / total * 100 if total > 0 else 0

    print()
    print("=" * 60)
    print(f"  OVERALL ACCURACY: {correct}/{total} = {accuracy:.1f}%")
    print("=" * 60)

    # Breakdown by domain and level if metadata available
    if task_metadata:
        # By level
        level_stats = {}
        domain_stats = {}
        for jr in judge_results:
            meta = task_metadata.get(jr["task_id"], {})
            # Level info is not in tasks.jsonl directly, but we can infer from metadata
            # For now just show overall + domain
            domain = meta.get("domain", "unknown")
            if domain not in domain_stats:
                domain_stats[domain] = {"correct": 0, "total": 0}
            domain_stats[domain]["total"] += 1
            if jr["correct"]:
                domain_stats[domain]["correct"] += 1

        if domain_stats:
            print()
            print("Per-Domain Accuracy:")
            print(f"  {'Domain':<40} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
            print("  " + "-" * 68)
            for domain in sorted(domain_stats.keys()):
                s = domain_stats[domain]
                acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
                print(f"  {domain:<40} {s['correct']:>8} {s['total']:>8} {acc:>9.1f}%")

    # Save CSV if requested
    if args.save_csv:
        csv_path = args.save_csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "correct", "accuracy", "reason", "answer", "ground_truth"])
            writer.writeheader()
            for jr in judge_results:
                writer.writerow({
                    "task_id": jr["task_id"],
                    "correct": jr["correct"],
                    "accuracy": 1.0 if jr["correct"] else 0.0,
                    "reason": jr["reason"],
                    "answer": jr["answer"][:200],
                    "ground_truth": jr["ground_truth"],
                })
        LOGGER.info(f"Per-task results saved to {csv_path}")

    print()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
