import copy
import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

import httpx
import tiktoken
from openai import (
    APIError,
    APIStatusError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from aiground.framework.thirdparty.openmanus.app.bedrock import BedrockClient
from aiground.framework.thirdparty.openmanus.app.config import LLMSettings, config
from aiground.framework.thirdparty.openmanus.app.exceptions import TokenLimitExceeded
from aiground.framework.thirdparty.openmanus.app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)
from .tracer import TraceMessage, Tracer

LOGGER = logging.getLogger(__name__)


def _should_retry_exception(exception: BaseException) -> bool:
    """
    Determine if an exception should trigger a retry.
    
    We should retry:
    - RateLimitError (429): Temporary, need to wait
    - Network/timeout errors: Temporary connection issues
    - Server errors (5xx): Temporary server issues
    
    We should NOT retry:
    - 400 Bad Request: Usually indicates malformed request, won't fix itself
    - 401/403: Authentication/authorization issues
    - TokenLimitExceeded: Our own limit, not fixable by retry
    - ValidationError: Request format issues
    """
    from aiground.framework.thirdparty.openmanus.app.exceptions import TokenLimitExceeded
    
    # Never retry token limit exceeded
    if isinstance(exception, TokenLimitExceeded):
        return False
    
    # Check for HTTP status code errors
    if isinstance(exception, APIStatusError):
        status_code = exception.status_code
        # Retry only for rate limit (429) and server errors (5xx)
        if status_code == 429:  # Rate limit
            LOGGER.info(f"Rate limit hit (429), will retry...")
            return True
        if status_code >= 500:  # Server errors
            LOGGER.info(f"Server error ({status_code}), will retry...")
            return True
        # Don't retry client errors (400, 401, 403, 404, etc.)
        LOGGER.warning(f"Client error ({status_code}), will NOT retry: {exception}")
        return False
    
    # For RateLimitError specifically (might not be caught by APIStatusError)
    if isinstance(exception, RateLimitError):
        LOGGER.info(f"Rate limit error, will retry...")
        return True
    
    # For other OpenAI errors, check if it's a retryable situation
    if isinstance(exception, OpenAIError):
        error_str = str(exception).lower()
        # Network/connection errors are retryable
        if any(term in error_str for term in ['timeout', 'connection', 'network']):
            LOGGER.info(f"Network error, will retry: {exception}")
            return True
        # Other OpenAI errors - don't retry by default
        LOGGER.warning(f"OpenAI error, will NOT retry: {exception}")
        return False
    
    # For generic exceptions, be conservative - don't retry
    LOGGER.warning(f"Unknown exception type {type(exception).__name__}, will NOT retry: {exception}")
    return False


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for debugging."""
    attempt = retry_state.attempt_number
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    LOGGER.warning(f"LLM API retry attempt {attempt}/6: {type(exception).__name__ if exception else 'unknown'}: {exception}")


REASONING_MODELS = ["o1", "o3-mini", "gpt-5.2", "gpt-5", "gpt-5.1","gpt-5-mini","gpt-5-nano"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-4-5-sonnet-20250929",
    "qwen3-vl-235b-a22b-instruct",
    "qwen3-vl-235b-a22b-thinking",
    "kimi-k2.5",
    # Gemini models (OpenAI-compatible endpoints) that support image inputs
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-2.5-flash-lite",
    "qwen2.5-vl-32b-instruct",
    "qwen3-omni-30b-a3b-instruct"
]

# Text-only models (no vision support)
TEXT_ONLY_MODELS = [
    "glm-4.7",
    "glm-5",
    "minimax-m2.5"
]


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    # _instances: Dict[str, "LLM"] = {}
    # def __new__(
    #     cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    # ):
    #     if config_name not in cls._instances:
    #         instance = super().__new__(cls)
    #         instance.__init__(config_name, llm_config)
    #         cls._instances[config_name] = instance
    #     return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, None)
            if llm_config is None:
                raise ValueError(f"No LLM configuration found for '{config_name}'")
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # Add token counting related attributes
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )
            self._last_usage: Optional[dict] = None

            # Max images per turn (None = unlimited)
            self.max_images_per_turn: Optional[int] = None

            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # If the model is not in tiktoken's presets, use cl100k_base as default
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            self.client = self._create_client()

            self.token_counter = TokenCounter(self.tokenizer)

    def _create_client(self):
        # Configure httpx client with proper timeouts for LLM requests
        # Connection timeout: 60 seconds (allow time for network/DNS resolution)
        # Read timeout: 600 seconds (LLM responses can take a long time, especially for complex tasks)
        # Write timeout: 120 seconds (for sending large payloads with images)
        # Pool timeout: 60 seconds (waiting for a connection from the pool)
        timeout_config = httpx.Timeout(
            connect=60.0,
            read=600.0,
            write=120.0,
            pool=60.0
        )
        
        # Configure connection limits for better concurrency handling
        # max_connections: maximum number of connections in the pool
        # max_keepalive_connections: connections to keep alive for reuse
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0
        )
        
        http_client = httpx.AsyncClient(
            timeout=timeout_config,
            limits=limits
        )
        
        # Add Claude-specific headers for prompt cache sticky routing
        default_headers = {}
        is_claude = "claude" in self.model.lower()
        if is_claude:
            # Use token-based sticky routing for Claude models to ensure prompt cache hits
            default_headers["Venus-Sticky-Routing"] = "token"
        
        if self.api_type == "azure":
            client = AsyncAzureOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                api_version=self.api_version,
                http_client=http_client,
                default_headers=default_headers if default_headers else None,
            )
        elif self.api_type == "aws":
            client = BedrockClient()
        else:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
                default_headers=default_headers if default_headers else None,
            )
        return client

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        LOGGER.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def _extract_usage_details(self, usage: Any) -> dict:
        if usage is None:
            return {}

        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "to_dict"):
            usage = usage.to_dict()

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0

        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or 0)
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                cached_tokens = int(prompt_details.get("cached_tokens") or prompt_details.get("cache_read_tokens") or 0)
        else:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None:
                cached_tokens = int(
                    getattr(prompt_details, "cached_tokens", 0)
                    or getattr(prompt_details, "cache_read_tokens", 0)
                    or 0
                )

        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        cached_tokens = max(0, cached_tokens)
        uncached_tokens = max(0, prompt_tokens - cached_tokens)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_cached": cached_tokens,
            "prompt_tokens_uncached": uncached_tokens,
        }

    def _set_last_usage(self, usage: Any) -> None:
        details = self._extract_usage_details(usage)
        if details:
            self._last_usage = details

    def consume_last_usage(self) -> Optional[dict]:
        usage = self._last_usage
        self._last_usage = None
        return usage

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False, model: str = "",
        max_images: Optional[int] = None,
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        import base64
        
        formatted_messages = []
        all_images = []  # Track all images with their message index

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images:
                    images_to_add = []
                    
                    # Handle base64_images (list of images)
                    if message.get("base64_images"):
                        images_to_add.extend(message["base64_images"])
                        del message["base64_images"]
                    
                    # Handle base64_image (single image) - for backward compatibility
                    if message.get("base64_image"):
                        images_to_add.append(message["base64_image"])
                        del message["base64_image"]
                    
                    if images_to_add:
                        # Initialize or convert content to appropriate format
                        if not message.get("content"):
                            message["content"] = []
                        elif isinstance(message["content"], str):
                            message["content"] = [
                                {"type": "text", "text": message["content"]}
                            ]
                        elif isinstance(message["content"], list):
                            # Convert string items to proper text objects
                            message["content"] = [
                                (
                                    {"type": "text", "text": item}
                                    if isinstance(item, str)
                                    else item
                                )
                                for item in message["content"]
                            ]

                        # Add all images to content
                        for img_b64 in images_to_add:
                            image_item = {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            }
                            message["content"].append(image_item)
                            # Track image with its message index and size
                            all_images.append({
                                "message_index": len(formatted_messages),
                                "image": image_item,
                                "size": len(img_b64) * 3 // 4  # Approximate base64 decoded size in bytes
                            })
                # If model doesn't support images but message has images, handle gracefully
                elif not supports_images:
                    if message.get("base64_image"):
                        del message["base64_image"]
                    if message.get("base64_images"):
                        del message["base64_images"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Enforce 7MB total image size limit (from oldest to newest)
        # When exceeded, drop images until we reach 4MB to avoid frequent cache invalidation
        MAX_TOTAL_IMAGE_SIZE = 7 * 1024 * 1024  # 7MB in bytes
        TARGET_IMAGE_SIZE = 4 * 1024 * 1024  # 4MB target after dropping
        
        if all_images:
            total_size = sum(img["size"] for img in all_images)
            
            if total_size > MAX_TOTAL_IMAGE_SIZE:
                # Remove oldest images until we're under the TARGET size (not just MAX)
                # This avoids frequent image dropping and preserves prompt cache
                images_to_remove = []
                current_size = total_size
                
                for img_info in all_images:
                    if current_size <= TARGET_IMAGE_SIZE:
                        break
                    images_to_remove.append(img_info)
                    current_size -= img_info["size"]
                
                # Remove images from their respective messages
                for img_info in images_to_remove:
                    msg_idx = img_info["message_index"]
                    msg = formatted_messages[msg_idx]
                    if isinstance(msg.get("content"), list):
                        # Remove this specific image from the message
                        msg["content"] = [
                            item for item in msg["content"]
                            if item != img_info["image"]
                        ]
                
                LOGGER.info(f"Removed {len(images_to_remove)} oldest images to stay within limit (was {total_size / 1024 / 1024:.2f}MB, now {current_size / 1024 / 1024:.2f}MB, target: {TARGET_IMAGE_SIZE / 1024 / 1024:.2f}MB)")

        # Enforce max_images limit: keep only the newest N images, drop oldest first
        if max_images is not None and all_images:
            # Rebuild the live image list (some may have been removed by size limit above)
            live_images = []
            for img_info in all_images:
                msg_idx = img_info["message_index"]
                msg = formatted_messages[msg_idx]
                if isinstance(msg.get("content"), list) and img_info["image"] in msg["content"]:
                    live_images.append(img_info)

            if len(live_images) > max_images:
                # Drop oldest images (beginning of the list) to keep only max_images newest
                num_to_drop = len(live_images) - max_images
                images_to_drop = live_images[:num_to_drop]
                for img_info in images_to_drop:
                    msg_idx = img_info["message_index"]
                    msg = formatted_messages[msg_idx]
                    if isinstance(msg.get("content"), list):
                        msg["content"] = [
                            item for item in msg["content"]
                            if item != img_info["image"]
                        ]
                LOGGER.info(f"Dropped {num_to_drop} oldest images to enforce max_images={max_images} (had {len(live_images)})")
        
        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception(_should_retry_exception),
        before_sleep=_log_retry_attempt,
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response
            response_format (dict): Optional response format specification. `{'type': 'json_object'}` for JSON response.

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images, self.model)
                messages = system_msgs + self.format_messages(
                    messages, supports_images, self.model,
                    max_images=self.max_images_per_turn,
                )
            else:
                messages = self.format_messages(
                    messages, supports_images, self.model,
                    max_images=self.max_images_per_turn,
                )

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )
            if response_format:
                params["response_format"] = response_format

            if not stream:
                # Non-streaming request
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )
                self._set_last_usage(response.usage)

                return response.choices[0].message.content

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens)

            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            LOGGER.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            LOGGER.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            LOGGER.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                LOGGER.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                LOGGER.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                LOGGER.error(f"API error: {oe}")
            raise
        except Exception:
            LOGGER.exception(f"Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception(_should_retry_exception),
        before_sleep=_log_retry_attempt,
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(
                messages, supports_images=True, model=self.model,
                max_images=self.max_images_per_turn,
            )

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content if isinstance(content, list) else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True, model=self.model,
                                        max_images=self.max_images_per_turn)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                self._set_last_usage(response.usage)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            LOGGER.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            LOGGER.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                LOGGER.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                LOGGER.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                LOGGER.error(f"API error: {oe}")
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception(_should_retry_exception),
        before_sleep=_log_retry_attempt,
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        tracer: Optional[Tracer] = None,
        **kwargs,
    ) -> Optional[ChatCompletionMessage]:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images, self.model)
                messages = system_msgs + self.format_messages(
                    messages, supports_images, self.model,
                    max_images=self.max_images_per_turn,
                )
            else:
                messages = self.format_messages(
                    messages, supports_images, self.model,
                    max_images=self.max_images_per_turn,
                )

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            is_claude = "claude" in self.model.lower()
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")
                
                # Add cache_control to the last tool for Claude models
                if is_claude and len(tools) > 0:
                    last_tool = tools[-1]
                    if "function" in last_tool:
                        last_tool["function"]["cache_control"] = {"type": "ephemeral"}
            
            # Add cache_control to system messages for Claude models
            if is_claude and system_msgs and len(system_msgs) > 0:
                last_system_msg = system_msgs[-1]
                if isinstance(last_system_msg.get("content"), str):
                    last_system_msg["content"] = [
                        {
                            "type": "text",
                            "text": last_system_msg["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                elif isinstance(last_system_msg.get("content"), list) and last_system_msg["content"]:
                    last_content_item = last_system_msg["content"][-1]
                    if isinstance(last_content_item, dict):
                        last_content_item["cache_control"] = {"type": "ephemeral"}

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }
            if len(messages) == 1:
                role = messages[0]["role"]
                if role != "user":  # If only one message and it's not user
                    xmessages = copy.deepcopy(messages)
                    xmessages[0]["role"] = "user"
                    params["messages"] = xmessages
            else:
                params["stream"] = True

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # LOGGER.debug(f"LLM request params: {params}")
            params["stream"] = False  # Always use non-streaming for tool requests
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )
            if tracer:
                trace_msg = TraceMessage(
                    req=params,
                    rsp=response,
                )
                tracer.trace(trace_msg)

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                LOGGER.warning("Invalid or empty response from LLM, retrying...")
                raise ValueError("Invalid or empty response from LLM")
            
            message = response.choices[0].message
            
            # Check for hallucinated tools or malformed content
            # Case 1: Hallucinated tools in tool_calls
            if message.tool_calls and tools:
                valid_tool_names = {t["function"]["name"] for t in tools}
                for tool_call in message.tool_calls:
                    if tool_call.function.name not in valid_tool_names:
                        err_msg = f"Hallucinated tool detected: {tool_call.function.name}"
                        LOGGER.warning(f"{err_msg}, retrying...")
                        raise ValueError(err_msg)

            # Case 2: Malformed content with "default_api:" or similar patterns
            if message.content:
                # Check for "default_api:" combined with "{" which looks like a tool call in text
                if "default_api:" in message.content and "{" in message.content:
                    err_msg = "Detected raw tool call in content (hallucination format)"
                    LOGGER.warning(f"{err_msg}, retrying...")
                    raise ValueError(err_msg)

            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )
            self._set_last_usage(response.usage)

            return message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            # Let tenacity handle retries for ValueError
            LOGGER.warning(f"Validation error in ask_tool (will retry): {ve}")
            raise
        except OpenAIError as oe:
            LOGGER.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                LOGGER.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                LOGGER.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                LOGGER.error(f"API error: {oe}")
            raise
        except Exception as e:
            LOGGER.error(f"Unexpected error in ask_tool: {e}")
            raise
