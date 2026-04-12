#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval.py — WebForge-Bench evaluation entry point.

Usage:
    # Run a single task
    python run_eval.py --config config.yaml --task-file tasks.jsonl --task-id 004771d2422a4915 --website-dir ./websites

    # Run all tasks
    python run_eval.py --config config.yaml --task-file tasks.jsonl --website-dir ./websites

    # Override output directory
    python run_eval.py --config config.yaml --task-file tasks.jsonl --website-dir ./websites --output-dir ./my_output
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import subprocess
from pathlib import Path

import yaml


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("webforge-eval")


def load_config(config_path: str) -> dict:
    """Load and validate YAML configuration file."""
    if not os.path.exists(config_path):
        LOGGER.error(f"Config file not found: {config_path}")
        LOGGER.error("Create one by copying the example: cp config.example.yaml config.yaml")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # --- Validate required fields ---
    llm_cfg = raw.get("llm", {})
    if not llm_cfg.get("model"):
        LOGGER.error("Config error: llm.model is required")
        sys.exit(1)
    if not llm_cfg.get("api_key") or llm_cfg["api_key"] == "your-api-key-here":
        LOGGER.error("Config error: llm.api_key must be set to a real key")
        sys.exit(1)

    # --- Build internal config dict (matching runner expectations) ---
    agent_cfg = raw.get("agent", {})
    image_cfg = raw.get("image", {})
    pipeline_cfg = raw.get("pipeline", {})

    config = {
        "llm": {
            "model": llm_cfg["model"],
            "api_key": llm_cfg["api_key"],
            "base_url": llm_cfg.get("base_url", ""),
            "api_type": llm_cfg.get("api_type", "Openai"),
            "max_tokens": llm_cfg.get("max_tokens", 40960),
            "temperature": llm_cfg.get("temperature", 1.0),
        },
        "browser": {
            "headless": False,
            "disable_security": True,
        },
        "image": {
            "use_image": image_cfg.get("use_image", True),
            "resize_width": image_cfg.get("resize_width", 1024),
            "max_height": image_cfg.get("max_height", 2048),
            "viewport_expansion": image_cfg.get("viewport_expansion", 0),
        },
        "max_steps": agent_cfg.get("max_steps", 50),
        "llm_timeout_seconds": agent_cfg.get("llm_timeout_seconds", 600),
        "llm_retry_count": agent_cfg.get("llm_retry_count", 10),
        "llm_retry_wait_seconds": agent_cfg.get("llm_retry_wait_seconds", 10),
    }

    # Optional: max_images_per_turn
    max_img = image_cfg.get("max_images_per_turn")
    if max_img is not None:
        config["image"]["max_images_per_turn"] = int(max_img)

    # Metadata (not passed to runner, used by run_eval.py itself)
    config["_agent_type"] = agent_cfg.get("type", "full")
    config["_save_trace"] = agent_cfg.get("save_trace", True)
    config["_max_workers"] = pipeline_cfg.get("max_workers", 1)

    return config


def load_tasks(task_file: str, task_id: str = None) -> list:
    """Load tasks from JSONL file."""
    tasks = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            if task_id and task["id"] != task_id:
                continue
            tasks.append(task)
    if task_id and not tasks:
        LOGGER.error(f"Task ID '{task_id}' not found in {task_file}")
        sys.exit(1)
    return tasks


def start_http_server(website_dir: str, port: int = 8000) -> subprocess.Popen:
    """Start a simple HTTP server to serve task websites."""
    LOGGER.info(f"Starting HTTP server on port {port} serving {website_dir}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", website_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import time
    time.sleep(1)
    return proc


async def run_single_task(config: dict, task: dict, base_url: str, output_dir: Path):
    """Run a single evaluation task."""
    task_id = task["id"]
    task_prompt = task["task_prompt"]
    ground_truth = task.get("ground_truth", "")
    url_path = task.get("url", f"/{task_id}/index.html")
    start_url = f"{base_url}{url_path}"

    task_output_dir = output_dir / task_id

    # Skip if result already exists
    result_path = task_output_dir / "result.json"
    if result_path.exists():
        LOGGER.info(f"[{task_id}] Already completed, skipping.")
        return True

    LOGGER.info(f"[{task_id}] Starting task...")

    agent_type = config.get("_agent_type", "full")
    save_trace = config.get("_save_trace", True)

    if agent_type == "simple":
        from agents.browser_use_simple_agent.web_browser_use_simple_runner import WebBrowserUseSimpleRunner
        runner = WebBrowserUseSimpleRunner(config=config, save_trace=save_trace)
    else:
        from agents.browser_use_agent.web_browser_use_runner import WebBrowserUseRunner
        runner = WebBrowserUseRunner(config=config, save_trace=save_trace)

    success = await runner.run(
        task=task_prompt,
        start_url=start_url,
        output_dir=task_output_dir,
        task_prompt=task_prompt,
        ground_truth=ground_truth,
    )
    LOGGER.info(f"[{task_id}] Completed: {'SUCCESS' if success else 'FAILED'}")
    return success


async def main():
    parser = argparse.ArgumentParser(
        description="WebForge-Bench Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Quick test with one task
  python run_eval.py --config config.yaml --task-file tasks.jsonl --task-id 010551772146e359 --website-dir ./websites

  # Full benchmark evaluation
  python run_eval.py --config config.yaml --task-file tasks.jsonl --website-dir ./websites
""",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file (see config.example.yaml)")
    parser.add_argument("--task-file", required=True, help="Path to tasks.jsonl")
    parser.add_argument("--task-id", default=None, help="Run a specific task by ID (default: run all)")
    parser.add_argument("--website-dir", required=True, help="Path to the websites/ directory")
    parser.add_argument("--output-dir", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load tasks
    tasks = load_tasks(args.task_file, args.task_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_type = config["_agent_type"]
    model_name = config["llm"]["model"]
    LOGGER.info(f"Loaded {len(tasks)} task(s) | agent={agent_type} | model={model_name}")
    LOGGER.info(f"  max_steps={config['max_steps']} | use_image={config['image']['use_image']} | "
                f"resize={config['image']['resize_width']}x{config['image']['max_height']}")

    # Start HTTP server
    server_proc = start_http_server(args.website_dir, args.port)
    base_url = f"http://localhost:{args.port}"

    try:
        completed = 0
        failed = 0
        for i, task in enumerate(tasks):
            LOGGER.info(f"--- Task {i+1}/{len(tasks)} ---")
            try:
                success = await run_single_task(config, task, base_url, output_dir)
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                LOGGER.exception(f"Task {task['id']} failed with error: {e}")
                failed += 1

        LOGGER.info(f"=== Done: {completed} completed, {failed} failed, {len(tasks)} total ===")
    finally:
        server_proc.terminate()
        server_proc.wait()
        LOGGER.info("HTTP server stopped.")


if __name__ == "__main__":
    asyncio.run(main())
