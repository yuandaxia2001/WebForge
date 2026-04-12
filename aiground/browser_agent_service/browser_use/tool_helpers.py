# -*- coding: utf-8 -*-
"""
Tool helper classes for web validation agent
"""

import json
import os
import datetime
from typing import Any, List, TYPE_CHECKING, Optional, Callable, Literal, Dict
from pydantic import BaseModel, Field
from aiground.framework.thirdparty.openmanus.app.tool.base import ToolResult
from .base_tool import BaseToolFunction

if TYPE_CHECKING:
    from .web_generation_tool import BrowserUseRequest


class ToolFunctionList:
    """Manages a collection of tool functions for the agent."""
    
    def __init__(self, tools: List[BaseToolFunction]):
        self.tools = {tool.name: tool for tool in tools}
    
    def to_param(self) -> List[dict]:
        """Convert all tools to OpenAI function calling format."""
        result = []
        for tool in self.tools.values():
            param_dict = tool.to_param()
            # Generate parameters from request_class if not provided
            if tool.parameters:
                param_dict["function"]["parameters"] = tool.parameters
            elif tool.request_class:
                param_dict["function"]["parameters"] = tool.request_class.model_json_schema()
            result.append(param_dict)
        return result
    
    def has(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self.tools
    
    async def execute(self, name: str, args: dict, session_id: str) -> ToolResult:
        """Execute a tool by name."""
        if name not in self.tools:
            return ToolResult(error=f"Unknown tool: {name}")
        
        tool = self.tools[name]
        try:
            return await tool.call(args, session_id)
        except Exception as e:
            return ToolResult(error=f"Tool execution failed: {str(e)}")


# Request models for tools

class RecordStepRequest(BaseModel):
    """Request model for record_step tool."""
    observation: str = Field(
        description="What you currently observe on the page (visible elements, content, current state)"
    )
    reasoning: str = Field(
        description="Your analysis of the current situation and planning for the next action"
    )
    action: str = Field(
        description="Description of the specific action you plan to execute next"
    )


class TerminateRequest(BaseModel):
    """Request model for terminate tool."""
    success: bool = Field(
        description="Whether the task was completed successfully. True = success, False = failure"
    )
    answer: str = Field(
        description="The final answer if success=True, or the failure reason if success=False"
    )


class ClearStorageAndNavigateRequest(BaseModel):
    """Request model for clear_storage_and_navigate tool."""
    confirm: bool = Field(
        default=True,
        description="Confirm clearing localStorage and sessionStorage, then navigate to start_url"
    )


# Browser use request for parameter schema
class BrowserUseRequestSchema(BaseModel):
    """Schema for browser_use tool parameters."""
    action: Literal[
        "go_to_url",
        "click_element",
        "input_text",
        "scroll_down",
        "scroll_up",
        "scroll_to_text",
        "send_keys",
        "get_dropdown_options",
        "select_dropdown_option",
        "go_back",
        "wait",
        "extract_content",
        "switch_tab",
        "open_tab",
        "close_tab",
    ] = Field(description="The browser action to perform")
    url: Optional[str] = Field(default=None, description="URL for 'go_to_url' or 'open_tab' actions")
    index: Optional[int] = Field(default=None, description="Element index for click/input actions")
    text: Optional[str] = Field(default=None, description="Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions")
    scroll_amount: Optional[int] = Field(default=None, description="Pixels to scroll for 'scroll_down' or 'scroll_up' actions")
    tab_id: Optional[int] = Field(default=None, description="Tab ID for 'switch_tab' action")
    goal: Optional[str] = Field(default=None, description="Extraction goal for 'extract_content' action")
    keys: Optional[str] = Field(default=None, description="Keys to send for keyboard actions")
    seconds: Optional[int] = Field(default=None, description="Seconds to wait for 'wait' action")


# Description constant for browser use tool
_BROWSER_USE_DESCRIPTION = (
    "Control a browser session to navigate, interact with web pages, "
    "click elements, input text, and extract information.\n\n"
    "**Actions:**\n"
    "- go_to_url: Navigate to a URL (requires 'url')\n"
    "- click_element: Click on an element (requires 'index')\n"
    "- input_text: Type text into an input field (requires 'index' and 'text')\n"
    "- scroll_down/scroll_up: Scroll the page (requires 'scroll_amount')\n"
    "- scroll_to_text: Scroll to specific text (requires 'text')\n"
    "- go_back: Navigate back in history\n"
    "- wait: Wait for seconds (requires 'seconds')\n"
    "- switch_tab: Switch to a tab (requires 'tab_id')\n"
    "- open_tab: Open new tab (requires 'url')\n"
    "- close_tab: Close current tab\n"
    "- extract_content: Extract page content (requires 'goal')\n"
    "- send_keys: Send keyboard keys (requires 'keys')"
)


class BrowserUseToolFunction(BaseToolFunction):
    """Wrapper for browser use tool."""
    name: str = "browser_use"
    description: str = _BROWSER_USE_DESCRIPTION
    request_class: type = BrowserUseRequestSchema
    _tool_func_impl: Optional[Callable] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, tool_func: Callable = None, **kwargs):
        super().__init__(
            name="browser_use",
            description=_BROWSER_USE_DESCRIPTION,
            request_class=BrowserUseRequestSchema,
            **kwargs
        )
        self._tool_func_impl = tool_func
    
    def set_tool_func(self, tool_func: Callable):
        """Set the tool function."""
        self._tool_func_impl = tool_func
    
    async def call(self, args: dict, session_id: str) -> ToolResult:
        """Execute browser action."""
        if self._tool_func_impl:
            return await self._tool_func_impl(args, session_id)
        return ToolResult(error="Browser tool function not set")


# Description for record_step tool
_RECORD_STEP_DESCRIPTION = (
    "Record your observation, reasoning, and planned action at each step. "
    "**YOU MUST CALL THIS TOOL BEFORE EVERY OTHER TOOL CALL.**\n\n"
    "Parameters:\n"
    "- observation: What you currently observe on the page\n"
    "- reasoning: Your analysis and planning for the next action\n"
    "- action: The specific action you plan to execute next"
)


class RecordStepToolFunction(BaseToolFunction):
    """Tool to record observation, reasoning, and action at each step."""
    name: str = "record_step"
    description: str = _RECORD_STEP_DESCRIPTION
    request_class: type = RecordStepRequest
    _steps: List[Dict] = []
    _step_count: int = 0
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(
            name="record_step",
            description=_RECORD_STEP_DESCRIPTION,
            request_class=RecordStepRequest,
            **kwargs
        )
        self._steps = []
        self._step_count = 0
    
    def reset(self):
        """Reset steps and step count for a new session."""
        self._steps = []
        self._step_count = 0
    
    def get_steps(self) -> List[Dict]:
        """Get all recorded steps."""
        return self._steps.copy()
    
    async def call(self, args: dict, session_id: str) -> ToolResult:
        """Record the step and save to internal list."""
        try:
            self._step_count += 1
            
            observation = args.get("observation", "")
            reasoning = args.get("reasoning", "")
            action = args.get("action", "")
            
            # Create record
            record = {
                "step": self._step_count,
                "timestamp": datetime.datetime.now().isoformat(),
                "observation": observation,
                "reasoning": reasoning,
                "action": action
            }
            
            # Save to internal list
            self._steps.append(record)
            
            return ToolResult(
                output=f"✅ Step {self._step_count} recorded.\n"
                       f"Observation: {observation[:100]}{'...' if len(observation) > 100 else ''}\n"
                       f"Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}\n"
                       f"Planned Action: {action}"
            )
        except Exception as e:
            return ToolResult(error=f"Failed to record step: {str(e)}")


# Description for clear_storage tool
_CLEAR_STORAGE_DESCRIPTION = (
    "Clear all browser localStorage and sessionStorage data, then navigate back to the start URL. "
    "Use this tool to reset the browser state if needed.\n\n"
    "⚠️ **IMPORTANT**: This tool can ONLY be used a MAXIMUM of 3 times!"
)

# Maximum allowed uses for clear_storage_and_navigate
_CLEAR_STORAGE_MAX_USES = 3


class ClearStorageToolFunction(BaseToolFunction):
    """Tool to clear browser storage and navigate to start URL."""
    name: str = "clear_storage_and_navigate"
    description: str = _CLEAR_STORAGE_DESCRIPTION
    request_class: type = ClearStorageAndNavigateRequest
    _browser_tool: Any = None
    _start_url: str = ""
    _usage_count: int = 0
    _max_uses: int = _CLEAR_STORAGE_MAX_USES
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(
            name="clear_storage_and_navigate",
            description=_CLEAR_STORAGE_DESCRIPTION,
            request_class=ClearStorageAndNavigateRequest,
            **kwargs
        )
        self._browser_tool = None
        self._start_url = ""
        self._usage_count = 0
        self._max_uses = _CLEAR_STORAGE_MAX_USES
    
    def set_browser_tool(self, browser_tool: Any):
        self._browser_tool = browser_tool
    
    def set_start_url(self, start_url: str):
        self._start_url = start_url
    
    def reset_usage_count(self):
        """Reset usage count for a new validation session."""
        self._usage_count = 0
    
    def get_remaining_uses(self) -> int:
        """Get the number of remaining uses."""
        return max(0, self._max_uses - self._usage_count)
    
    async def call(self, args: dict, session_id: str) -> ToolResult:
        """Clear storage and navigate to start URL."""
        if not self._browser_tool:
            return ToolResult(error="Browser tool not initialized")
        if not self._start_url:
            return ToolResult(error="Start URL not set")
        
        # Check usage limit
        if self._usage_count >= self._max_uses:
            return ToolResult(
                error=f"❌ USAGE LIMIT REACHED: clear_storage_and_navigate has already been used {self._max_uses} times. "
                      f"No more uses allowed!"
            )
        
        try:
            # Get browser state
            browser_state = await self._browser_tool._browser_store.get_resource(session_id)
            if not browser_state or not browser_state.context:
                return ToolResult(error="Browser context not found")
            
            ctx = browser_state.context
            page = await ctx.get_current_page()
            
            # Clear localStorage and sessionStorage
            await page.evaluate("""
                () => {
                    localStorage.clear();
                    sessionStorage.clear();
                    console.log('Storage cleared');
                }
            """)
            
            # Navigate to start URL
            await page.goto(self._start_url)
            try:
                await asyncio.wait_for(page.wait_for_load_state("domcontentloaded"), timeout=5.0)
            except asyncio.TimeoutError:
                logger.debug("Page load state timeout (5s) after clear_and_navigate, continuing anyway")
            except Exception as e:
                logger.debug(f"Page load state check failed after clear_and_navigate: {e}, continuing anyway")
            
            # Increment usage counter
            self._usage_count += 1
            remaining = self._max_uses - self._usage_count
            
            return ToolResult(
                output=f"✅ Successfully cleared storage and navigated to: {self._start_url}\n"
                       f"📊 Uses: {self._usage_count}/{self._max_uses} | Remaining: {remaining}"
            )
        except Exception as e:
            return ToolResult(error=f"Failed to clear storage and navigate: {str(e)}")


# Description for terminate tool
_TERMINATE_DESCRIPTION = (
    "End the task solving session. Call this when you have completed the task or cannot proceed further.\n\n"
    "**Parameters:**\n"
    "- `success` (boolean, required): True if the task was completed successfully, False if it failed\n"
    "- `answer` (string, required): Provide the final answer if success=True, or the failure reason if success=False.\n\n"
    "**Examples:**\n"
    "Success: {\"success\": true, \"answer\": \"Final answer content\"}\n"
    "Failure: {\"success\": false, \"answer\": \"Could not find the submit button on the page\"}"
)


class TerminateToolFunction(BaseToolFunction):
    """Tool to terminate the task solving session."""
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    request_class: type = TerminateRequest
    _callback: Optional[Callable] = None
    _last_result: Optional[Dict] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, callback: Callable = None, **kwargs):
        super().__init__(
            name="terminate",
            description=_TERMINATE_DESCRIPTION,
            request_class=TerminateRequest,
            **kwargs
        )
        self._callback = callback
        self._last_result = None
    
    def set_callback(self, callback: Callable):
        """Set a callback to be called when terminate is invoked."""
        self._callback = callback
    
    def get_last_result(self) -> Optional[Dict]:
        """Get the last terminate result."""
        return self._last_result
    
    def reset(self):
        """Reset the last result."""
        self._last_result = None
    
    async def call(self, args: dict, session_id: str) -> ToolResult:
        """Terminate the session."""
        try:
            success = args.get("success", True)
            answer = args.get("answer", "")

            if not answer:
                return ToolResult(
                    error="❌ answer is required. Provide the final answer or the failure reason."
                )

            # Store the result
            self._last_result = {
                "success": success,
                "answer": answer
            }

            # Call callback if provided
            if self._callback:
                await self._callback(self._last_result)

            if success:
                return ToolResult(
                    output=f"✅ Task completed successfully! Answer: {answer}\nSession terminated."
                )
            return ToolResult(
                output=f"❌ Task failed. Reason: {answer}\nSession terminated."
            )
        except Exception as e:
            return ToolResult(error=f"Failed to terminate: {str(e)}")
