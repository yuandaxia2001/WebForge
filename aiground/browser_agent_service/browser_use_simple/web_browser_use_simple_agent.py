# -*- coding: utf-8 -*-
"""
web_browser_use_simple_agent

Simplified agent for solving tasks on benchmark websites using browser tools.
This version removes record_step requirement.
Designed for smaller models.
"""

import asyncio
import base64
import copy
import datetime
import io
import json
import logging
import os
import typing
import uuid
from typing import Any, List, Optional, Callable

from PIL import Image
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)
from pydantic import BaseModel, Field

from aiground.framework.thirdparty.openmanus.app.agent.toolcall import (
    TOOL_CALL_REQUIRED,
)
from aiground.framework.thirdparty.openmanus.app.exceptions import TokenLimitExceeded
from aiground.framework.thirdparty.openmanus.app.schema import (
    ROLE_TYPE,
    Memory,
    Message,
    ToolCall,
    ToolChoice,
)
from aiground.framework.thirdparty.openmanus.app.tool.base import ToolResult
from aiground.framework.thirdparty.openmanus.app.tracer import Tracer

# Reuse tool helpers from original browser_use
from aiground.browser_agent_service.browser_use.tool_helpers import (
    ToolFunctionList,
    BrowserUseToolFunction,
    TerminateToolFunction,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aiground.browser_agent_service.browser_use.web_generation_tool import BrowserUseTool, BrowserUseRequest
else:
    BrowserUseTool = Any
    BrowserUseRequest = Any

LOGGER = logging.getLogger(__name__)


class WebBrowserUseSimpleTaskRequest(BaseModel):
    """Request model for Web Browser Use Simple Agent task execution."""
    task: str = Field(description="The task description to complete")
    max_steps: int = Field(default=50, description="Maximum number of steps to execute")
    start_url: str = Field(description="The start URL for the website to solve")
    output_dir: str = Field(description="Directory to save task outputs")
    task_prompt: str = Field(default="", description="Original task prompt from tasks.jsonl")
    ground_truth: str = Field(default="", description="Ground truth answer from tasks.jsonl")


def load_system_prompt():
    # Load Browser Use Simple Agent specific prompt
    prompt_path = os.path.join(os.path.dirname(__file__), "../../prompts/browser_use_simple/system_prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        LOGGER.warning(f"System prompt file not found at {prompt_path}, using default.")
        return "You are a Browser Use Agent. Complete the task using the browser tools."

TASK_EXECUTOR_SYSTEM_PROMPT = load_system_prompt()


BROWSER_NEXT_STEP_PROMPT = """
[Current Browser State]
--------------------------------------------------
**URL:** {url_placeholder}
**Title:** {title_placeholder}
**Tabs:** {tabs_placeholder}

**Interactive Elements:**
{elements_placeholder}
--------------------------------------------------

**Last Action Results:**
{results_placeholder}
"""


class ReasoningResult(BaseModel):
    should_act: bool = Field(default=False, description="Whether the agent should act")
    tool_calls: Optional[List[ChatCompletionMessageToolCallUnion]] = Field(
        default=None, description="Tool calls"
    )
    content: str = Field(description="Content", default="")


class WebBrowserUseSimpleAgent:
    def __init__(
        self,
        tool: "BrowserUseTool",
        trace_data_dir: Optional[str] = None,
        llm_retry_count: int = 10,
        llm_retry_wait_seconds: int = 10
    ):
        self._tool = tool
        self._trace_data_dir = trace_data_dir
        self._llm_retry_count = llm_retry_count
        self._llm_retry_wait_seconds = llm_retry_wait_seconds

    def init_asyncio_loop(self):
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    def async_run(self, task):
        self.init_asyncio_loop()
        asyncio.run(task)

    async def execute(self, request: "WebBrowserUseSimpleTaskRequest"):
        task_executor = WebBrowserUseSimpleTaskExecutor(
            question=request.task,
            tool=self._tool,
            trace_data_dir=self._trace_data_dir,
            start_url=request.start_url,
            output_dir=request.output_dir,
            max_steps=request.max_steps,
            llm_retry_count=self._llm_retry_count,
            llm_retry_wait_seconds=self._llm_retry_wait_seconds,
            task_prompt=request.task_prompt,
            ground_truth=request.ground_truth
        )
        await task_executor.initialize_session()

        async def generator():
            step_cnt = 0
            max_turns = request.max_steps
            LOGGER.info(f"[GENERATOR] Starting generator, max_turns={max_turns}")
            while True:
                LOGGER.info(f"[GENERATOR] Loop iteration {step_cnt}, is_finished={task_executor.is_finished()}")
                if step_cnt >= max_turns:
                    task_executor._max_steps_exceeded = True
                    task_executor._finish_reason = "max_steps_exceeded"
                    await task_executor._save_result_json()
                    msg = await task_executor.final_execute()
                    yield "data: " + json.dumps({"code": 201, "message": "success", "data": msg}, ensure_ascii=False) + "\n\n"
                    break
                if task_executor.is_finished():
                    msg = task_executor.final_result()
                    yield "data: " + json.dumps({"code": 200, "message": "success", "data": msg}, ensure_ascii=False) + "\n\n"
                    break
                step_cnt += 1
                LOGGER.info(f"[GENERATOR] Creating task for execute() step {step_cnt}")
                task = asyncio.create_task(task_executor.execute())
                LOGGER.info(f"[GENERATOR] Task created, waiting for messages from message queue")
                while True:
                    msg = await task_executor._message_queue.get()
                    LOGGER.info(f"[GENERATOR] Got message from queue: {msg}")
                    if msg == "STEP_DONE":
                        break
                    yield "data: " + json.dumps({"code": 100, "message": "success", "data": msg}, ensure_ascii=False) + "\n\n"
                LOGGER.info(f"[GENERATOR] STEP_DONE received, awaiting task completion")
                await task

        async def cleanup(cancelled: bool = False):
            if cancelled:
                task_executor._cancelled = True
            await task_executor.cleanup_session()

        return generator(), cleanup


class WebBrowserUseSimpleTaskExecutor:
    def __init__(
        self,
        question: str,
        tool: "BrowserUseTool",
        trace_data_dir: Optional[str] = None,
        start_url: str = "",
        output_dir: str = "",
        max_steps: int = 50,
        llm_retry_count: int = 10,
        llm_retry_wait_seconds: int = 10,
        task_prompt: str = "",
        ground_truth: str = ""
    ):
        self.name = "TaskSolvingAgent"
        self._question = question
        self._tool = tool
        self._llm = self._tool.get_llm()
        self.memory = Memory()
        self._session_id = uuid.uuid4().hex
        self._trace_data_dir = trace_data_dir
        self._max_steps = max_steps
        self._max_steps_exceeded = False
        self._start_url = start_url
        self._output_dir = output_dir
        self._llm_retry_count = llm_retry_count
        self._llm_retry_wait_seconds = llm_retry_wait_seconds
        
        self._result_json_path = os.path.join(output_dir, "result.json")
        self._cancelled = False
        self._terminate_result = None
        
        # Initialize tools - NO record_step, only browser_use and terminate
        self._terminate_tool = TerminateToolFunction()
        self._terminate_tool.set_callback(self._on_terminate)
        
        self._browser_use_tool_func = BrowserUseToolFunction()
        self._browser_use_tool_func.set_tool_func(self._browser_use_execute)
        
        # Simple version: only browser_use and terminate (no record_step)
        self._available_tools = ToolFunctionList(
            tools=[
                self._browser_use_tool_func,
                self._terminate_tool,
            ]
        )
        
        self._tool_choice = ToolChoice.AUTO
        self._current_system_prompt = TASK_EXECUTOR_SYSTEM_PROMPT
        self._current_step = 0
        self._finish_reason = "unknown"
        self._last_action_results: str = ""
        self._tracer: Optional[Tracer] = None
        self._trace_request_dir: Optional[str] = None
        self._trace_file_path: Optional[str] = None
        self._trace_images_dir: Optional[str] = None

        self._task_prompt: str = task_prompt
        self._ground_truth: str = ground_truth
        self._start_time: float = 0.0
        self._end_time: float = 0.0

        # Steps recording - collect from LLM content instead of record_step tool
        self._steps: list = []

        # Statistics
        self._stats = {
            "total_steps": 0,
            "total_actions": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_cached": 0,
                "prompt_tokens_uncached": 0,
            }
        }

    async def _on_terminate(self, result: dict):
        """Callback when terminate is called."""
        self._terminate_result = result
        import time
        self._end_time = time.time()
        await self._save_result_json()
    
    async def _save_result_json(self):
        """Save the final result to a JSON file."""
        try:
            if self._end_time == 0.0:
                import time
                self._end_time = time.time()
            
            success = True
            answer = ""
            if self._terminate_result:
                success = self._terminate_result.get("success", True)
                answer = self._terminate_result.get("answer", "")
            elif self._finish_reason == "max_steps_exceeded" or self._max_steps_exceeded:
                success = False
                answer = "Reached the maximum number of steps"
            elif self._finish_reason == "token_limit":
                success = False
                answer = "Token limit exceeded"
            elif self._finish_reason == "unexpected_error":
                success = False
                answer = "Task ended unexpectedly due to an error"
            elif self._finish_reason not in ["terminated", "success"]:
                success = False
                answer = f"Task ended with reason: {self._finish_reason}"

            elapsed_time = self._end_time - self._start_time if self._end_time > 0 and self._start_time > 0 else 0.0

            result = {
                "task_prompt": self._task_prompt,
                "ground_truth": self._ground_truth,
                "success": success,
                "answer": answer,
                "steps": self._steps,
                "llm_turns": self._current_step,
                "max_steps": self._max_steps,
                "elapsed_time_seconds": round(elapsed_time, 2),
                "stats": self._stats
            }
            
            os.makedirs(self._output_dir, exist_ok=True)
            
            with open(self._result_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            LOGGER.info(f"Result saved to {self._result_json_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save result JSON: {e}")

    async def _browser_use_execute(self, args: dict, session_id: str) -> ToolResult:
        """Execute browser use tool."""
        from aiground.browser_agent_service.browser_use.web_generation_tool import BrowserUseRequest
        
        try:
            self._stats["total_actions"] += 1
            req = BrowserUseRequest(**args)
            result = await self._tool.execute(req, session_id)
            return result
        except Exception as e:
            return ToolResult(error=f"Browser action failed: {str(e)}")

    def _resolve_trace_dirs(self) -> None:
        self._trace_request_dir = None
        self._trace_file_path = None
        self._trace_images_dir = None

        root = (self._trace_data_dir or "").strip()
        if not root:
            return
        self._trace_request_dir = os.path.join(root, self._session_id)
        self._trace_file_path = os.path.join(self._trace_request_dir, "trace.jsonl")
        self._trace_images_dir = os.path.join(self._trace_request_dir, "images")

    def _process_image(self, b64_img: str) -> str:
        if not b64_img:
            return b64_img
        try:
            target_width = getattr(self._tool._property, "image_resize_width", 1024)
            max_height = getattr(self._tool._property, "image_max_height", 10240)
            img_data = base64.b64decode(b64_img)
            img = Image.open(io.BytesIO(img_data))
            w, h = img.size
            if w and target_width and w != target_width:
                ratio = float(target_width) / float(w)
                new_h = int(h * ratio)
                img = img.resize((int(target_width), int(new_h)))
            else:
                new_h = h
            if max_height and new_h > max_height:
                img = img.crop((0, 0, int(target_width), int(max_height)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            LOGGER.error("Image processing failed: %s", e)
            return b64_img

    def _clean_memory(self):
        pass

    async def format_next_step_user_prompt(self, session_id: str):
        state_rsp = await self._tool.get_current_state(session_id)
        if state_rsp.error:
            return
        browser_state: dict = json.loads(state_rsp.output)
        if not browser_state or browser_state.get("error"):
            return

        url = browser_state.get("url", "N/A")
        title = browser_state.get("title", "N/A")
        tabs = browser_state.get("tabs", [])
        if tabs:
            max_tabs = 8
            tab_lines = []
            for t in tabs[:max_tabs]:
                page_id = t.get("page_id", "N/A")
                tab_url = t.get("url", "N/A")
                tab_title = t.get("title", "N/A")
                current_mark = " *current*" if tab_url == url else ""
                tab_lines.append(f"- [{page_id}]{current_mark} {tab_title} — {tab_url}")
            more = f"\n... (+{len(tabs) - max_tabs} more)" if len(tabs) > max_tabs else ""
            tabs_info = f"{len(tabs)} tabs open\n" + "\n".join(tab_lines) + more
        else:
            tabs_info = "No other tabs"

        interactive_elements = browser_state.get("interactive_elements", "No interactive elements found")
        results_info = self._last_action_results or ""
        max_len = 2000
        if len(results_info) > max_len:
            results_info = results_info[:max_len] + "\n...[truncated]..."
        content = BROWSER_NEXT_STEP_PROMPT.format(
            url_placeholder=url,
            title_placeholder=title,
            tabs_placeholder=tabs_info,
            elements_placeholder=interactive_elements,
            results_placeholder=results_info,
        )
        
        # Simple reminder
        content += "\n\n---\n**Reminder:** Use `browser_use` to interact with the page, or `terminate` to end the task."
        
        b64_img = None
        if state_rsp.base64_image and self._tool._property.use_image:
            b64_img = self._process_image(state_rsp.base64_image)
        user_message = Message.user_message(content=content, base64_image=b64_img)
        self.memory.add_message(user_message)

    async def _gather_initial_context(self):
        """Gathers initial context including task and start URL."""
        context_msg = "### Task Context ###\n\n"
        context_msg += f"#### Task:\n{self._question}\n\n"
        context_msg += f"### Start URL ###\n{self._start_url}\n\n"
        context_msg += "### Instructions ###\n"
        context_msg += "- Use `browser_use` to interact with the page\n"
        context_msg += "- Call `terminate` when the task is completed, providing the final answer or failure reason\n\n"
        context_msg += "Work toward completing the task and report the final answer in `terminate`.\n"

        self.memory.add_message(Message.user_message(context_msg))

    async def initialize_session(self):
        import time
        self._start_time = time.time()
        
        await self._tool.create_session(self._session_id)
        self._finished = False
        self._current_system_prompt = TASK_EXECUTOR_SYSTEM_PROMPT
        self._message_queue = asyncio.Queue()
        self._last_content = ""
        self._resolve_trace_dirs()
        
        os.makedirs(self._output_dir, exist_ok=True)
        
        if self._trace_file_path and self._trace_images_dir:
            self._tracer = Tracer(file_path=self._trace_file_path, images_dir=self._trace_images_dir).init(mode="a")
            self._tracer.trace(
                {
                    "event": "request_start",
                    "request_id": self._session_id,
                    "question": self._question,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )
        
        # Navigate to start URL
        from aiground.browser_agent_service.browser_use.web_generation_tool import BrowserUseRequest
        req = BrowserUseRequest(action="go_to_url", url=self._start_url)
        await self._tool.execute(req, self._session_id)
        
        await self._gather_initial_context()

    def is_finished(self) -> bool:
        return self._finished

    def final_result(self) -> str:
        if self._finish_reason == "unknown":
            self._finish_reason = "success"
        return self._last_content

    async def cleanup_session(self):
        if self._cancelled:
            LOGGER.info("Task was cancelled, skipping result.json save for retry")
            if os.path.exists(self._result_json_path):
                try:
                    os.remove(self._result_json_path)
                    LOGGER.info(f"Removed incomplete result.json: {self._result_json_path}")
                except Exception as e:
                    LOGGER.warning(f"Failed to remove incomplete result.json: {e}")
        else:
            if not os.path.exists(self._result_json_path):
                if self._finish_reason == "unknown":
                    self._finish_reason = "unexpected_error"
                    LOGGER.warning(f"Task ended with unknown finish_reason at step {self._current_step}/{self._max_steps}, marking as unexpected_error")
                await self._save_result_json()
        
        await self._tool.destroy_session(self._session_id)
        if self._tracer:
            try:
                self._tracer.trace(
                    {
                        "event": "request_end",
                        "request_id": self._session_id,
                        "finish_reason": self._finish_reason,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                )
            except Exception:
                pass
            self._tracer.exit()

    async def put_message(self, msg):
        await self._message_queue.put(msg)

    async def execute(self):
        try:
            self._current_step += 1
            self._stats["total_steps"] = self._current_step
            LOGGER.info(f"[EXECUTE] Starting step {self._current_step}")
            self._clean_memory()
            
            await self.format_next_step_user_prompt(self._session_id)
            
            think_result = await self._reasoning()
            LOGGER.info(f"[EXECUTE] _reasoning() returned, should_act={think_result.should_act}, tool_calls_count={len(think_result.tool_calls) if think_result.tool_calls else 0}")
            
            if not self._last_content:
                self._last_content = think_result.content
            if not think_result.should_act:
                LOGGER.info(f"[EXECUTE] No action needed, returning")
                self._last_content = think_result.content
                return "Thinking complete - no action needed"
            
            LOGGER.info(f"[EXECUTE] Calling _acting() with {len(think_result.tool_calls) if think_result.tool_calls else 0} tool calls")
            result = await self._acting(think_result)
            LOGGER.info(f"[EXECUTE] _acting() completed, result length: {len(result) if result else 0}")
            return result
        except Exception as e:
            LOGGER.exception(f"[EXECUTE] Execution error in web_browser_use_simple_agent: {e}")
            await self.put_message(f"Error during execution: {str(e)}")
            raise e
        finally:
            LOGGER.info(f"[EXECUTE] Step {self._current_step} done, sending STEP_DONE")
            await self.put_message("STEP_DONE")

    async def final_execute(self):
        tool_rsp = await self._tool.get_current_content(self._session_id)
        return tool_rsp.output if tool_rsp.output else tool_rsp.error

    async def _reasoning(self) -> ReasoningResult:
        ret = ReasoningResult(should_act=False)
        
        last_error = None
        response = None
        for attempt in range(self._llm_retry_count):
            try:
                def _strip_unsupported_schema_fields(obj: Any, keys_to_remove: set) -> Any:
                    if isinstance(obj, dict):
                        new_obj = {}
                        for k, v in obj.items():
                            if k in keys_to_remove:
                                continue
                            new_obj[k] = _strip_unsupported_schema_fields(v, keys_to_remove)
                        return new_obj
                    if isinstance(obj, list):
                        return [_strip_unsupported_schema_fields(x, keys_to_remove) for x in obj]
                    return obj

                tools_payload = self._available_tools.to_param()
                model_name = str(getattr(self._llm, "model", "") or "").lower()
                if "gemini" in model_name:
                    tools_payload = _strip_unsupported_schema_fields(copy.deepcopy(tools_payload), keys_to_remove={"dependencies"})

                response = await self._llm.ask_tool(
                    self.memory.get_recent_messages(10000),
                    system_msgs=([Message.system_message(self._current_system_prompt)] if self._current_system_prompt else None),
                    tools=tools_payload,
                    tool_choice=self._tool_choice,
                    tracer=self._tracer,
                    timeout=self._tool.get_llm_timeout_seconds(),
                )
                LOGGER.info(f"[REASONING] LLM response received")
                LOGGER.info(f"[REASONING] has_tool_calls={bool(response and response.tool_calls)}, tool_calls_count={len(response.tool_calls) if response and response.tool_calls else 0}")
                
                raw_content = response.content if response and response.content else ""
                
                # Handle case where content is a list instead of string
                if isinstance(raw_content, list):
                    LOGGER.warning(f"Model returned list content instead of string: {raw_content}")
                    text_parts = []
                    for item in raw_content:
                        if isinstance(item, dict) and "text" in item:
                            text_val = item["text"]
                            if isinstance(text_val, str) and not text_val.startswith("<ctrl"):
                                text_parts.append(text_val)
                        elif isinstance(item, str) and not item.startswith("<ctrl"):
                            text_parts.append(item)
                    raw_content = " ".join(text_parts) if text_parts else ""

                # Update token usage stats
                try:
                    usage_dict = self._llm.consume_last_usage()
                    if usage_dict:
                        p_tokens = int(usage_dict.get("prompt_tokens", 0) or 0)
                        c_tokens = int(usage_dict.get("completion_tokens", 0) or 0)
                        t_tokens = int(usage_dict.get("total_tokens", 0) or 0)
                        cached = int(usage_dict.get("prompt_tokens_cached", 0) or 0)
                        uncached = int(usage_dict.get("prompt_tokens_uncached", 0) or 0)
                        
                        self._stats["token_usage"]["prompt_tokens"] += p_tokens
                        self._stats["token_usage"]["completion_tokens"] += c_tokens
                        self._stats["token_usage"]["total_tokens"] += t_tokens
                        self._stats["token_usage"]["prompt_tokens_cached"] += cached
                        self._stats["token_usage"]["prompt_tokens_uncached"] += uncached
                        
                        LOGGER.info(f"[REASONING] Token usage updated: prompt={p_tokens}, completion={c_tokens}, total={t_tokens}")
                    else:
                        LOGGER.warning(f"[REASONING] No usage information available from LLM")
                except Exception as e:
                    LOGGER.warning(f"[REASONING] Failed to extract token usage: {e}")
                break
                
            except Exception as e:
                last_error = e
                if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                    token_limit_error = e.__cause__
                    await self.put_message(f"Token limit error: {token_limit_error}")
                    self.memory.add_message(Message.assistant_message(f"Maximum token limit reached: {str(token_limit_error)}"))
                    self._finish_reason = "token_limit"
                    self._finished = True
                    ret.should_act = False
                    return ret
                
                if attempt < self._llm_retry_count - 1:
                    await self.put_message(f"LLM request failed (attempt {attempt + 1}/{self._llm_retry_count}): {str(e)}")
                    await asyncio.sleep(self._llm_retry_wait_seconds)
                else:
                    await self.put_message(f"All {self._llm_retry_count} LLM retries exhausted. Last error: {str(e)}")
                    ret.should_act = False
                    ret.content = f"LLM temporarily unavailable (retried {self._llm_retry_count} times). Continuing..."
                    self.memory.add_message(Message.assistant_message(ret.content))
                    return ret
        
        if last_error and not response:
            await self.put_message(f"LLM request failed with no response: {str(last_error)}")
            ret.should_act = False
            ret.content = f"LLM error: {str(last_error)}"
            self.memory.add_message(Message.assistant_message(ret.content))
            return ret

        ret.content = raw_content
        ret.tool_calls = response.tool_calls if response and response.tool_calls else []
        LOGGER.info(f"[REASONING] Set ret.tool_calls count={len(ret.tool_calls)}")
        await self.put_message(f"Agent's thoughts: {ret.content}")
        await self.put_message(f"Agent selected {len(ret.tool_calls) if ret.tool_calls else 0} tools to use")

        # Record the step from LLM content (since we don't have record_step tool)
        content_str = raw_content if isinstance(raw_content, str) else str(raw_content or "")
        if content_str.strip():
            self._steps.append({
                "step": self._current_step,
                "timestamp": datetime.datetime.now().isoformat(),
                "content": content_str.strip()[:500],
                "tool_calls": [tc.function.name for tc in ret.tool_calls] if ret.tool_calls else []
            })

        try:
            assistant_msg = (
                Message.from_tool_calls(content=ret.content, tool_calls=ret.tool_calls)
                if ret.tool_calls
                else Message.assistant_message(ret.content)
            )
        except Exception as e:
            LOGGER.error(f"[REASONING] Failed to create assistant message: {e}", exc_info=True)
            assistant_msg = Message.assistant_message(ret.content if isinstance(ret.content, str) else "")

        model_name = str(getattr(self._llm, "model", "") or "").lower()
        if "gemini" in model_name:
            thought_sig = None
            if hasattr(response, "thought_signature"):
                 thought_sig = response.thought_signature
            elif hasattr(response, "model_extra") and response.model_extra and "thought_signature" in response.model_extra:
                 thought_sig = response.model_extra.get("thought_signature")
            elif hasattr(response, "__dict__") and "thought_signature" in response.__dict__:
                 thought_sig = response.__dict__["thought_signature"]

            if thought_sig:
                assistant_msg.thought_signature = thought_sig
            
        self.memory.add_message(assistant_msg)

        if self._tool_choice == ToolChoice.REQUIRED and not ret.tool_calls:
            ret.should_act = True
            return ret
        if self._tool_choice == ToolChoice.AUTO and not ret.tool_calls:
            ret.should_act = bool(ret.content)
            return ret
        ret.should_act = bool(ret.tool_calls)
        return ret

    async def _acting(self, thinking: ReasoningResult):
        if not thinking.tool_calls:
            if self._tool_choice == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)
            if self._finish_reason == "unknown":
                self._finish_reason = "no_tools_called"
            return self.memory.messages[-1].content or "No content or commands to execute"

        results = []
        tool_calls = thinking.tool_calls or []
        if len(tool_calls) > 1:
            await self.put_message(f"Executing {len(tool_calls)} tool calls in sequence...")

        results_info_parts = []
        previous_action_failed = False
        previous_error_message = ""
        
        for command in tool_calls:
            if previous_action_failed:
                skip_message = f"Action '{command.function.name}' SKIPPED: Previous action failed with error: {previous_error_message}"
                await self.put_message(skip_message)
                
                tool_msg = Message.tool_message(
                    content=skip_message,
                    tool_call_id=command.id,
                    name=command.function.name,
                )
                self.memory.add_message(tool_msg)
                results_info_parts.append(f"Tool: {command.function.name}\nOutput: {skip_message}")
                results.append(skip_message)
                continue
            
            result = await self._execute_tool(command)
            tool_output = result.output if result.output else result.error
            results_info_parts.append(f"Tool: {command.function.name}\nOutput: {tool_output}")
            
            await self.put_message(f"Tool '{command.function.name}' completed! Result: {tool_output[:200]}...")

            if result.error:
                previous_action_failed = True
                previous_error_message = result.error
                LOGGER.warning(f"Action '{command.function.name}' failed: {result.error}")

            if result.base64_images:
                tool_msg = Message.tool_message(
                    content=tool_output,
                    tool_call_id=command.id,
                    name=command.function.name,
                    base64_images=result.base64_images,
                )
            else:
                tool_msg = Message.tool_message(
                    content=tool_output,
                    tool_call_id=command.id,
                    name=command.function.name,
                    base64_image=result.base64_image,
                )
            self.memory.add_message(tool_msg)
            results.append(tool_output)

        self._last_action_results = "\n\n".join(results_info_parts)
        tool_name_set = set([call.function.name for call in thinking.tool_calls])
        content_str = thinking.content if isinstance(thinking.content, str) else str(thinking.content or "")
        content_str = content_str.strip()
        
        if len(tool_name_set) == 0:
            self._finished = True
            self._finish_reason = "no_tools_called"
            if content_str:
                self._last_content = content_str
            return "Thinking complete - no action needed"
        
        if content_str:
            self._last_content = content_str

        if "terminate" in tool_name_set:
            self._finished = True
            self._finish_reason = "terminated"
        
        return "\n\n".join(results)

    async def _execute_tool(self, command: ToolCall) -> ToolResult:
        if not command or not command.function or not command.function.name:
            return ToolResult(error="Error: Invalid command format")
        name = command.function.name
        
        # Handle browser_use specially
        if name == "browser_use":
            try:
                args = json.loads(command.function.arguments or "{}")
                await self.put_message(f"Activating tool: '{name}'...")
                result = await self._browser_use_execute(args, self._session_id)
                if result.error:
                    observation = f"Tool '{name}' encountered a problem: {result.error}"
                    return ToolResult(
                        output=observation,
                        error=result.error,
                        base64_image=result.base64_image,
                        base64_images=result.base64_images
                    )
                else:
                    observation = f"Observed output of cmd `{name}` executed:\n{result.output}" if result.output else f"Cmd `{name}` completed with no output"
                return ToolResult(
                    output=observation, 
                    base64_image=result.base64_image,
                    base64_images=result.base64_images
                )
            except json.JSONDecodeError as je:
                return ToolResult(error=f"Invalid JSON arguments: {str(je)}")
            except Exception as e:
                return ToolResult(error=f"Tool '{name}' failed: {str(e)}")
        
        if not self._available_tools.has(name):
            return ToolResult(error=f"Error: Unknown tool '{name}'. Please use one of the available tools.")
        try:
            args = json.loads(command.function.arguments or "{}")
            await self.put_message(f"Activating tool: '{name}'...")
            result: ToolResult = await self._available_tools.execute(name=name, args=args, session_id=self._session_id)
            if result.error:
                observation = f"Tool '{name}' encountered a problem: {result.error}"
                return ToolResult(
                    output=observation,
                    error=result.error,
                    base64_image=result.base64_image,
                    base64_images=result.base64_images
                )
            else:
                observation = f"Observed output of cmd `{name}` executed:\n{result.output}" if result.output else f"Cmd `{name}` completed with no output"
            return ToolResult(
                output=observation, 
                base64_image=result.base64_image,
                base64_images=result.base64_images
            )
        except json.JSONDecodeError as je:
            error_msg = f"Tool '{name}' has invalid JSON arguments: {str(je)}"
            LOGGER.exception(f"JSON decode error for tool '{name}': {je}")
            return ToolResult(error=error_msg)
        except Exception as e:
            error_msg = f"Tool '{name}' encountered a problem: {str(e)}"
            LOGGER.exception(error_msg)
            return ToolResult(error=f"Error: {error_msg}")

    def _update_memory(
        self,
        role: ROLE_TYPE,
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }
        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))
