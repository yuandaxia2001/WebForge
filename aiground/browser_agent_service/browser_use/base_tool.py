import typing
from typing import Any, Optional
from abc import ABC
from pydantic import BaseModel
from aiground.framework.thirdparty.openmanus.app.tool.base import ToolResult

class BaseToolFunction(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    request_class: typing.Type[BaseModel] = None
    tool_func: Optional[typing.Callable[[BaseModel, str], typing.Coroutine[Any, Any, ToolResult]]] = None

    class Config:
        arbitrary_types_allowed = True

    def to_param(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def call(self, args: dict, session_id: str) -> ToolResult:
        if self.tool_func is None:
            return ToolResult(error=f"Tool function not set for {self.name}")
        req = self.request_class.model_validate(args)
        return await self.tool_func(req, session_id)

