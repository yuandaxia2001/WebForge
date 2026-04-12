import logging
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from aiground.framework.thirdparty.openmanus.app.agent.browser import (
    BrowserContextHelper,
)
from aiground.framework.thirdparty.openmanus.app.agent.toolcall import ToolCallAgent
from aiground.framework.thirdparty.openmanus.app.config import MCPServerConfig, config
from aiground.framework.thirdparty.openmanus.app.prompt.manus import (
    NEXT_STEP_PROMPT,
    SYSTEM_PROMPT,
)
from aiground.framework.thirdparty.openmanus.app.tool import Terminate, ToolCollection
from aiground.framework.thirdparty.openmanus.app.tool.ask_human import AskHuman
from aiground.framework.thirdparty.openmanus.app.tool.browser_use_tool import (
    BrowserUseTool,
)
from aiground.framework.thirdparty.openmanus.app.tool.mcp import (
    MCPClients,
    MCPClientTool,
)
from aiground.framework.thirdparty.openmanus.app.tool.python_execute import (
    PythonExecute,
)
from aiground.framework.thirdparty.openmanus.app.tool.str_replace_editor import (
    StrReplaceEditor,
)

LOGGER = logging.getLogger(__name__)


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """Initialize basic components synchronously."""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        await self.init_mcp_servers_with(config.mcp_config.servers)

    async def init_mcp_servers_with(self, servers: Dict[str, MCPServerConfig]) -> None:
        """Initialize connections to multiple MCP servers."""
        for server_id, server_config in servers.items():
            await self.init_one_mcp_server(server_id, server_config)
        self._initialized = True

    async def init_one_mcp_server(
        self, server_id: str, server_config: MCPServerConfig
    ) -> None:
        """Initialize a single MCP server."""
        try:
            if server_config.type == "sse":
                if server_config.url:
                    headers = None
                    if (
                        hasattr(server_config, "env")
                        and server_config.env
                        and server_config.env.get("token")
                    ):
                        token = server_config.env.get("token")
                        headers = {"Authorization": f"Bearer {token}"}
                    LOGGER.info(
                        f"Connecting to MCP server {server_id} via SSE, headers: {headers}",
                    )
                    await self.connect_mcp_server(
                        server_config.url, server_id, sse_headers=headers
                    )
                    LOGGER.info(
                        f"Connected to MCP server {server_id} at {server_config.url}"
                    )
            elif server_config.type == "stdio":
                if server_config.command:
                    await self.connect_mcp_server(
                        server_config.command,
                        server_id,
                        use_stdio=True,
                        stdio_args=server_config.args,
                    )
                    LOGGER.info(
                        f"Connected to MCP server {server_id} using command {server_config.command}"
                    )
        except Exception as e:
            LOGGER.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
        sse_headers: dict = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id, sse_headers)
            self.connected_servers[server_id or server_url] = server_url

        # Update available tools with only the new tools from this server
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # Rebuild available tools without the disconnected server's tools
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """Clean up Manus agent resources."""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # Disconnect from all MCP servers only if we were initialized
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result
