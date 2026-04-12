# -*- coding: utf-8 -*-
"""
Web Browser Use Simple Runner - Executes the simplified browser use agent.

This module provides a wrapper for running the simplified browser use agent
without record_step requirement.
Designed for smaller models.
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

LOGGER = logging.getLogger(__name__)

# Setup import paths to use local aiground
_PIPELINE_ROOT = Path(__file__).parent.parent.parent

def _setup_import_paths():
    """Add local aiground to Python path."""
    if str(_PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PIPELINE_ROOT))

_setup_import_paths()


class WebBrowserUseSimpleRunner:
    """
    Wrapper for running the simplified browser use agent.

    This class provides a clean interface for running
    browser-based benchmark tasks using the simplified agent
    without record_step requirement.
    """
    
    def __init__(self, config: Dict[str, Any], save_trace: bool = True):
        self.config = config
        self.save_trace = save_trace
    
    async def run(
        self,
        task: str,
        start_url: str,
        output_dir: Optional[Path] = None,
        task_prompt: str = "",
        ground_truth: str = "",
    ) -> bool:
        browser_tool = None
        cleanup = None
        
        try:
            # Import from local aiground - simple browser use agent
            from aiground.browser_agent_service.browser_use_simple.web_browser_use_simple_agent import (
                WebBrowserUseSimpleAgent,
                WebBrowserUseSimpleTaskRequest,
            )
            # Reuse BrowserUseTool and BrowserUseProperty from original browser_use
            from aiground.browser_agent_service.browser_use.web_generation_tool import (
                BrowserUseTool,
                BrowserUseProperty,
            )
            from aiground.framework.thirdparty.openmanus.app.llm import LLM
            from aiground.framework.thirdparty.openmanus.app.config import LLMSettings
            from aiground.framework.thirdparty.browser_use.browser.browser import (
                Browser as BrowserUseBrowser,
                BrowserConfig,
                ProxySettings,
            )
            
            if output_dir is None:
                output_dir = Path.cwd() / "output" / "task_solving"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize LLM
            llm_config = self.config.get('llm', {})
            llm_settings = LLMSettings(
                model=llm_config.get('model', 'gemini-3-flash'),
                api_key=llm_config.get('api_key', ''),
                base_url=llm_config.get('base_url', ''),
                api_type=llm_config.get('api_type', 'Openai'),
                api_version=llm_config.get('api_version', ''),
                max_tokens=llm_config.get('max_tokens', 40960),
                temperature=llm_config.get('temperature', 1.0),
            )
            llm_config_dict = {"browser_agent_llm": llm_settings}
            llm = LLM(config_name="browser_agent_llm", llm_config=llm_config_dict)
            
            # Set max images per turn if configured
            image_cfg = self.config.get('image', {})
            max_images_per_turn = image_cfg.get('max_images_per_turn')
            if max_images_per_turn is not None:
                llm.max_images_per_turn = int(max_images_per_turn)
            
            # Initialize browser
            browser_cfg = self.config.get('browser', {})
            proxy_cfg = browser_cfg.get('proxy', {})
            proxy = None
            if proxy_cfg and proxy_cfg.get('server'):
                proxy = ProxySettings(
                    server=proxy_cfg['server'],
                    username=proxy_cfg.get('username'),
                    password=proxy_cfg.get('password'),
                )
            
            browser_config = BrowserConfig(
                headless=browser_cfg.get('headless', False),
                disable_security=browser_cfg.get('disable_security', True),
                wss_url=browser_cfg.get('wss_url') or None,
                cdp_url=browser_cfg.get('cdp_url') or None,
                proxy=proxy,
                chrome_remote_debugging_port=None,  # Disable fixed port to allow parallel browser instances
            )
            
            browser = BrowserUseBrowser(config=browser_config)
            
            tool_property = BrowserUseProperty(
                browser=browser,
                llm=llm,
                llm_timeout_seconds=self.config.get('llm_timeout_seconds', 600),
                use_image=image_cfg.get('use_image', True),
                image_resize_width=image_cfg.get('resize_width', 1024),
                image_max_height=image_cfg.get('max_height', 2048),
                viewport_expansion=image_cfg.get('viewport_expansion', 0),
            )
            
            browser_tool = BrowserUseTool(property=tool_property)
            
            llm_retry_count = self.config.get('llm_retry_count', 10)
            llm_retry_wait = self.config.get('llm_retry_wait_seconds', 10)
            trace_dir = str(output_dir / "trace") if self.save_trace else None
            
            # Use SimpleAgent instead of the original
            agent = WebBrowserUseSimpleAgent(
                tool=browser_tool,
                trace_data_dir=trace_dir,
                llm_retry_count=llm_retry_count,
                llm_retry_wait_seconds=llm_retry_wait,
            )

            # Simplified task prompt - no record_step instructions
            full_task = f"""
Complete the task on the website at {start_url}

**Task:**
{task}

**Instructions:**
1. Observe the current page state and decide on the next action
2. Call `browser_use` to interact with the page, or `terminate` when done
3. Call `terminate` when the task is completed, providing the final answer or failure reason
"""

            max_steps = self.config.get('max_steps', 50)
            req = WebBrowserUseSimpleTaskRequest(
                task=full_task,
                max_steps=max_steps,
                start_url=start_url,
                output_dir=str(output_dir),
                task_prompt=task_prompt,
                ground_truth=ground_truth
            )

            LOGGER.info(f"Starting simple browser use task at: {start_url}")
            generator, cleanup = await agent.execute(req)
            
            success = False
            async for result in generator:
                try:
                    import yaml
                    data_str = result.replace("data: ", "").strip()
                    if not data_str:
                        continue
                    data_json = yaml.safe_load(data_str)
                    
                    if data_json['code'] == 100:
                        LOGGER.debug(f"Agent: {data_json['data'][:100]}...")
                    elif data_json['code'] == 200:
                        LOGGER.info(f"SUCCESS: {data_json['data'][:100]}...")
                        success = True
                    elif data_json['code'] == 201:
                        LOGGER.info(f"MAX STEPS REACHED: {data_json['data'][:100]}...")
                        success = True
                except Exception as e:
                    LOGGER.debug(f"Parse error (non-critical): {e}")
            
            result_path = output_dir / "result.json"
            if result_path.exists():
                success = True
                LOGGER.info(f"Simple browser use task completed: {result_path}")
            else:
                LOGGER.warning(f"Simple browser use result not found: {result_path}")
            
            return success
        
        except asyncio.CancelledError:
            LOGGER.warning("Simple browser use task was cancelled")
            if cleanup:
                try:
                    await cleanup(cancelled=True)
                    cleanup = None
                except Exception as e:
                    LOGGER.warning(f"Cleanup during cancellation warning: {e}")
            raise
        except Exception as e:
            LOGGER.exception(f"Simple browser use error: {e}")
            return False
        finally:
            if cleanup:
                try:
                    await cleanup(cancelled=False)
                except Exception as e:
                    LOGGER.warning(f"Session cleanup warning: {e}")
            
            if browser_tool:
                try:
                    await browser_tool._cleanup()
                    LOGGER.info("Browser tool cleanup completed")
                except Exception as e:
                    LOGGER.warning(f"Browser cleanup warning: {e}")
