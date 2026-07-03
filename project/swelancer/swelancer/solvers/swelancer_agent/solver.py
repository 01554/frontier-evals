import asyncio
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, AsyncGenerator, cast

import openai
import structlog
import tenacity
import tiktoken
from dotenv import load_dotenv
from nanoeval_alcatraz.alcatraz_computer_interface import (
    AlcatrazComputerRuntime,
)
from openai import AsyncOpenAI
from openai._types import NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
)
from typing_extensions import override

import chz
from nanoeval.contrib.utils import run_with_startup_timeout
from nanoeval.eval import RolloutSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerRuntime,
    RuntimeConfig,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    Step,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask
from swelancer.eval import SWELancerTask

logger = structlog.stdlib.get_logger(component=__name__)

STARTUP_TIMEOUT = 1200  # seconds (20 mins)
DEFAULT_CTX_LIMIT = 110000
MODEL_TO_CTX_LIMIT = {
    # llama.cpp server context varies by local quant/server launch. Keep a safety
    # margin for chat-template and tool-schema tokens that tiktoken does not count exactly.
    "kimi-k2.7-code-q2": 224000,
    "openai/kimi-k2.7-code-q2": 224000,
    "kimi-k2.7-code-q3": 84000,
    "openai/kimi-k2.7-code-q3": 84000,
    "openai/gpt-4o": 128000,
    "openai/gpt-4.1": 1047576,
    "openai/o1": 200000,
    "openai/o3": 200000,
    "openai/o3-mini": 200000,
    "openai/o4-mini": 200000,
    "anthropic/claude-3-5-sonnet": 200000,
    "anthropic/claude-3.5-sonnet-20240620": 200000,
    "deepseek/deepseek-coder": 128000,
    "meta-llama/llama-3-70b-instruct": 8000,
    "meta-llama/llama-3.3-70b-instruct": 128000,
}

load_dotenv()


PYTHON_TOOL_NAME = "python"
PYTHON_EXECUTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": PYTHON_TOOL_NAME,
        "description": (
            "Execute a complete, self-contained Python script in the SWE-Lancer "
            "task container. Use this to inspect files, edit files, and run commands."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "A complete Python script to run with python -c.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class ModelResponse:
    content: str
    message: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None


def get_client_and_model(model: str) -> tuple[AsyncOpenAI, str]:
    if model.startswith("openai/"):
        model = model[len("openai/") :]
        base_url = None
        api_key = os.environ.get("OPENAI_API_KEY")
    else:  # Otherwise, use openrouter
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY")

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    ), model


def count_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """
    Count the number of tokens in a list of messages. Uses tiktoken to count tokens.
    If the model is not found, use gpt-4.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4")

    num_tokens = 0
    for message in messages:
        # Every message follows format: {"role": role, "content": content}
        num_tokens += 4  # Every message follows format: <im_start>{role/name}\n{content}<im_end>\n
        for value in message.values():
            num_tokens += len(encoding.encode(str(value)))

    return num_tokens


def trim_messages(
    task: SWELancerTask, messages: list[dict[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Trim messages to fit within token limit by removing older messages."""
    ctx_logger = logger.bind(
        run_group_id=task.run_group_id,
        runs_dir=task.runs_dir,
        run_id=task.run_id,
    )

    max_tokens = MODEL_TO_CTX_LIMIT[model] if model in MODEL_TO_CTX_LIMIT else DEFAULT_CTX_LIMIT
    while len(messages) > 1 and count_tokens(messages, model) > max_tokens:
        ctx_logger.info("Trimming message turn...", destinations=["run"])
        messages.pop(1)
        while len(messages) > 1 and messages[1].get("role") == "tool":
            messages.pop(1)
    return messages


def _extract_markdown_python_blocks(text: str) -> list[str]:
    python_blocks = re.findall(
        r"```python\s*\n(.*?)\n```",
        text,
        re.DOTALL,
    )
    return [dedent(block) for block in python_blocks]


def _extract_tool_payload_python_blocks(tool_payload: Any) -> list[str]:
    blocks: list[str] = []
    if isinstance(tool_payload, dict):
        for key, value in tool_payload.items():
            if key == "code" and isinstance(value, str):
                blocks.append(dedent(value))
            elif isinstance(value, str):
                blocks.extend(_extract_markdown_python_blocks(value))
            else:
                blocks.extend(_extract_tool_payload_python_blocks(value))
    elif isinstance(tool_payload, list):
        for value in tool_payload:
            blocks.extend(_extract_tool_payload_python_blocks(value))
    return blocks


def extract_python_blocks(model_response: str) -> list[str]:
    python_blocks = _extract_markdown_python_blocks(model_response)
    if python_blocks:
        return python_blocks

    # Some OpenAI-compatible local models emit learned tool-call markup even when
    # no tools were supplied. Recover Python from common tool-call payload shapes.
    tool_blocks: list[str] = []
    marker = "<|tool_call_argument_begin|>"
    end_marker = "<|tool_call_end|>"
    decoder = json.JSONDecoder()
    search_start = 0
    while True:
        marker_start = model_response.find(marker, search_start)
        if marker_start == -1:
            break
        payload_start = marker_start + len(marker)
        payload_end = model_response.find(end_marker, payload_start)
        if payload_end == -1:
            payload_end = len(model_response)
        payload = model_response[payload_start:payload_end].strip()
        try:
            tool_args, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            search_start = payload_end
            continue
        tool_blocks.extend(_extract_tool_payload_python_blocks(tool_args))
        search_start = payload_end
    return tool_blocks


def _parse_tool_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments

    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def extract_python_tool_call_blocks(tool_calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for index, tool_call in enumerate(tool_calls):
        tool_call_id = str(tool_call.get("id") or f"python_call_{index}")
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = _parse_tool_arguments(function.get("arguments"))

        if name not in {PYTHON_TOOL_NAME, f"functions.{PYTHON_TOOL_NAME}", "execute_python"}:
            blocks.append(
                (
                    tool_call_id,
                    (
                        f"Error: unknown tool {name!r}. The only available tool is "
                        f"{PYTHON_TOOL_NAME!r} with a JSON argument named 'code'."
                    ),
                )
            )
            continue

        extracted = _extract_tool_payload_python_blocks(arguments)
        if extracted:
            blocks.append((tool_call_id, extracted[0]))
            continue

        if isinstance(arguments, str):
            fallback_blocks = _extract_markdown_python_blocks(arguments)
            if fallback_blocks:
                blocks.append((tool_call_id, fallback_blocks[0]))
            elif arguments.strip():
                blocks.append((tool_call_id, dedent(arguments)))
            else:
                blocks.append((tool_call_id, "Error: python tool call did not include code."))
            continue

        blocks.append((tool_call_id, "Error: python tool call did not include code."))
    return blocks


OPENROUTER_RETRY_EXCEPTIONS = (
    openai.BadRequestError,
    openai.InternalServerError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    json.JSONDecodeError,
)


async def get_model_response(
    task: SWELancerTask,
    messages: list[dict[str, Any]],
    model: str,
    reasoning_effort: str | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> ModelResponse:
    client, model = get_client_and_model(model)

    messages = trim_messages(task, messages, model)

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=1, max=300),  # Max wait time of 5 minutes
        # 20 attempts or 20 minutes max
        stop=(tenacity.stop_after_attempt(20) | tenacity.stop_after_delay(20 * 60)),
        retry=tenacity.retry_if_exception_type(OPENROUTER_RETRY_EXCEPTIONS),
        before_sleep=(tenacity.before_sleep_log(logger._logger, logging.INFO) if logger else None),
        reraise=True,
    )
    async def _get_model_response() -> ChatCompletion:
        chat_completion = await client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], messages),
            model=model,
            reasoning_effort=cast(ChatCompletionReasoningEffort, reasoning_effort)
            if reasoning_effort is not None
            else NotGiven(),
            tools=tools if tools is not None else NotGiven(),
        )
        if chat_completion is None:
            raise json.JSONDecodeError(msg="chat_completion is None", doc="", pos=0)
        if chat_completion.choices is None or len(chat_completion.choices) == 0:
            raise json.JSONDecodeError(
                msg="chat_completion.choices is None or empty", doc="", pos=0
            )
        if chat_completion.choices[0] is None:
            raise json.JSONDecodeError(msg="chat_completion.choices[0] is None", doc="", pos=0)
        if chat_completion.choices[0].message is None:
            raise json.JSONDecodeError(
                msg="chat_completion.choices[0].message is None", doc="", pos=0
            )
        return chat_completion

    chat_completion = await _get_model_response()

    # get number of input and output tokens
    input_tokens = chat_completion.usage.prompt_tokens if chat_completion.usage is not None else 0
    output_tokens = (
        chat_completion.usage.completion_tokens if chat_completion.usage is not None else 0
    )
    reasoning_tokens = (
        chat_completion.usage.completion_tokens_details.reasoning_tokens
        if chat_completion.usage is not None
        and chat_completion.usage.completion_tokens_details is not None
        else 0
    )
    message = chat_completion.choices[0].message
    message_dict = message.model_dump(exclude_none=True)
    tool_calls = cast(list[dict[str, Any]], message_dict.get("tool_calls") or [])
    return ModelResponse(
        content=message.content or "",
        message=message_dict,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


@chz.chz
class SimpleAgentSolver(PythonCodingSolver):
    computer_runtime: ComputerRuntime = chz.field(default_factory=AlcatrazComputerRuntime)

    name: str = chz.field(default="SimpleAgentSolver")
    model: str = chz.field(
        default="openai/gpt-4o",
        doc="Assumes model to be in OpenRouter format of PROVIDER/MODEL",
    )
    reasoning_effort: str | None = chz.field(default=None)
    model_response_timeout: float = chz.field(
        default=900.0,
        doc="Maximum wall-clock seconds to wait for one model response before grading.",
    )
    max_turns: int = chz.field(
        default=40,
        doc="Maximum model/code-execution turns before grading.",
    )
    max_rollout_seconds: float | None = chz.field(
        default=None,
        doc="Maximum wall-clock seconds for the full task rollout before grading.",
    )
    use_python_tool: bool = chz.field(
        default=False,
        doc="Expose a native OpenAI-compatible python(code) tool to the model.",
    )

    runtime_config: RuntimeConfig = chz.field(default_factory=RuntimeConfig)

    def shortname(self) -> str:
        return "simple-solver"

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        assert isinstance(task, SWELancerTask), (
            f"SimpleAgentSolver only supports SWELancerTasks, got {type(task)}"
        )

        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
        )

        try:
            ctx_logger.info("Starting computer...", destinations=["run"])
            async with run_with_startup_timeout(
                self.computer_runtime, task, STARTUP_TIMEOUT
            ) as computer:
                # 1. Query the API / some agent
                messages = []

                assert "content" in task.prompt[0]
                assert isinstance(task.prompt[0]["content"], str)
                agent_prompt = (
                    TOOL_AWARE_AGENT_PROMPT if self.use_python_tool else ADDITIONAL_AGENT_PROMPT
                )
                instructions = task.prompt[0]["content"] + "\n\n" + agent_prompt
                messages.append({"role": "user", "content": instructions})
                tools = [PYTHON_EXECUTION_TOOL] if self.use_python_tool else None
                rollout_deadline = (
                    time.monotonic() + self.max_rollout_seconds
                    if self.max_rollout_seconds is not None
                    else None
                )
                rollout_timeout_message = (
                    f"Rollout timed out after {self.max_rollout_seconds} seconds; grading current state."
                )

                for remaining_turns in range(self.max_turns, 0, -1):
                    response_timeout = self.model_response_timeout
                    if rollout_deadline is not None:
                        remaining_rollout_seconds = rollout_deadline - time.monotonic()
                        if remaining_rollout_seconds <= 0:
                            ctx_logger.info(rollout_timeout_message, destinations=["run"])
                            break
                        response_timeout = min(response_timeout, remaining_rollout_seconds)

                    try:
                        async with asyncio.timeout(response_timeout):
                            response = await get_model_response(
                                task,
                                messages,
                                self.model,
                                self.reasoning_effort,
                                ctx_logger,
                                tools,
                            )
                    except asyncio.TimeoutError:
                        if rollout_deadline is not None and time.monotonic() >= rollout_deadline:
                            ctx_logger.info(rollout_timeout_message, destinations=["run"])
                        else:
                            ctx_logger.info(
                                f"Model response timed out after {self.model_response_timeout} seconds; grading current state.",
                                destinations=["run"],
                            )
                        break
                    task.input_tokens += response.input_tokens
                    task.output_tokens += response.output_tokens
                    task.reasoning_tokens += (
                        response.reasoning_tokens if response.reasoning_tokens is not None else 0
                    )

                    model_response = response.content
                    if self.use_python_tool:
                        messages.append(response.message)
                    else:
                        messages.append({"role": "assistant", "content": model_response})

                    ctx_logger.info(
                        f"Message history on turn {remaining_turns}, number of messages: {len(messages)}: {messages}",
                        destinations=["run"],
                    )

                    execution_output = None

                    # Check for user-tool calls
                    timeout = 300  # 5 mins
                    if rollout_deadline is not None:
                        remaining_rollout_seconds = rollout_deadline - time.monotonic()
                        if remaining_rollout_seconds <= 0:
                            ctx_logger.info(rollout_timeout_message, destinations=["run"])
                            break
                        timeout = min(timeout, remaining_rollout_seconds)
                    try:
                        async with asyncio.timeout(timeout):
                            python_tool_blocks = (
                                extract_python_tool_call_blocks(response.tool_calls)
                                if self.use_python_tool
                                else []
                            )
                            if python_tool_blocks:
                                ctx_logger.info(
                                    "Python tool call detected",
                                    destinations=["run"],
                                )
                                for tool_call_id, code in python_tool_blocks:
                                    if code.startswith("Error: "):
                                        execution_output = code
                                    else:
                                        execution_res = await computer.send_shell_command(
                                            f"python -c {shlex.quote(code)}"
                                        )
                                        execution_output = execution_res.output.decode(
                                            "utf-8", errors="replace"
                                        )
                                    if execution_output is None or execution_output == "":
                                        execution_output = "No output or errors from the code. If you think you're done, send 'DONE' and we will grade your submission. Otherwise, continue working."
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call_id,
                                            "content": f"{execution_output}\nTurns left: {remaining_turns - 1}",
                                        }
                                    )
                                continue
                            if "<user-tool>" in model_response:
                                ctx_logger.info(
                                    "User tool call detected",
                                    destinations=["run"],
                                )
                                try:
                                    await computer.check_shell_command(
                                        "unzip -q -P 'secretpasswordhere' /app/tests.zip -d /app/tests"
                                    )
                                    await computer.check_shell_command(
                                        "bash -c -i 'user-tool'; exit"
                                    )
                                    execution_output = (
                                        "User tool output is in /app/expensify/user_tool"
                                    )
                                except Exception as e:
                                    execution_output = f"Error: {e}"
                                await computer.send_shell_command("rm -rf /app/tests")
                                await computer.send_shell_command("rm -rf /app/tests")
                            elif model_response.strip().lower() == "done":
                                break
                            else:
                                ctx_logger.info(
                                    "No user tool call detected; executing code",
                                    destinations=["run"],
                                )
                                python_blocks = extract_python_blocks(model_response)

                                if not python_blocks:
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": (
                                                "Warning: No Python code blocks were found in the response. "
                                                "There is no functions.* tool API available here, and learned "
                                                "tool-call markup such as <|tool_call_begin|> cannot be executed "
                                                "unless its JSON arguments contain a complete `code` string. "
                                                "Reply with exactly one fenced Python code block using "
                                                "```python ... ```."
                                            ),
                                        }
                                    )
                                    continue

                                code = python_blocks[0]

                                execution_res = await computer.send_shell_command(
                                    f"python -c {shlex.quote(code)}"
                                )
                                execution_output = execution_res.output.decode(
                                    "utf-8", errors="replace"
                                )
                    except asyncio.TimeoutError:
                        if rollout_deadline is not None and time.monotonic() >= rollout_deadline:
                            ctx_logger.info(rollout_timeout_message, destinations=["run"])
                            break
                        execution_output = (
                            f"Timeout: The code took too longer than {timeout} seconds to execute."
                        )
                        ctx_logger.info(
                            "Agent code timed out",
                            destinations=["run"],
                        )

                    if execution_output is None or execution_output == "":
                        execution_output = "No output or errors from the code. If you think you're done, send 'DONE' and we will grade your submission. Otherwise, continue working."

                    # Append the code and its output to the messages
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{execution_output}\nTurns left: {remaining_turns - 1}",
                        }
                    )

                # 2. Grade and yield the final result
                try:
                    grade = await task.grade(computer, self.runtime_config)
                except Exception as e:
                    raise RolloutSystemError(f"Error during grading: {e}") from e

                yield FinalResult(grade=grade)
        except asyncio.TimeoutError as e:
            ctx_logger.exception("Computer startup timed out", destinations=["run"])
            raise RolloutSystemError("Computer startup timed out") from e
        except Exception as e:
            if isinstance(e, RolloutSystemError):
                ctx_logger.exception(
                    f"RolloutSystemError in {task.run_id}: {e}", destinations=["run"]
                )
                raise
            ctx_logger.exception(
                f"Unexpected error in task {task.run_id}: {e}", destinations=["run"]
            )
            raise RolloutSystemError(f"Unexpected error during rollout: {e}") from e


@chz.chz
class ToolAwareAgentSolver(SimpleAgentSolver):
    name: str = chz.field(default="ToolAwareAgentSolver")
    use_python_tool: bool = chz.field(default=True)

    def shortname(self) -> str:
        return "tool-aware-solver"


ADDITIONAL_AGENT_PROMPT = (
    """The repository is cloned in your CWD. You must send Python code in backticks in each response to me, and I will execute the code and send you back the result, for example:

```python
import os
print(os.listdir())
```
Do not use function calls, tool calls, XML tags, JSON tool payloads, or markup such as <|tool_call_begin|>. There are no callable tools named functions.python, functions.ExpensifyCodebase, or similar. Your only executable interface is exactly one fenced Python code block.
You can include other content in your message, as long as it has the Python code block. You must only submit one code block per message.
The python code block is your means of interacting with the environment; write Python code to view files, edit files, and run commands. Do not leave any steps incomplete. You must implement all fixes yourself.
When you are done, send "DONE" (and nothing else - you don't need a python code block), and we will grade your submission.
Please note that the Python code is not a Jupyter notebook; you must write a full Python script each time, including imports, and printing things if you want to see the output.
"""
    + """
You may find it useful to search for particular keywords in the codebase using the `grep` command via subprocess:
```python
import subprocess
print(subprocess.check_output(['grep', '-irn', '"foo"', '/app/expensify"]))
```
where `"foo"` is the keyword you want to search for.

Remember, you can go step by step. If you are unsure about a command, use a step to look up help e.g.
```python
import subprocess
print(subprocess.check_output(['grep', '--help']))
```

You should start by browsing the files in the codebase to understand the problem and formulate a plan.
"""
)


TOOL_AWARE_AGENT_PROMPT = (
    """The repository is cloned in your CWD. You have a native function tool named `python`.
Use the `python` tool with a JSON argument named `code` whenever you need to inspect files, edit files, run commands, or test your changes.
The code must be a complete, self-contained Python script because each tool execution is invoked with `python -c`.
Do not call any tool except `python`. Do not emit raw tool-call markup manually. If your backend cannot emit a structured tool call, a single fenced ```python ... ``` block is accepted as a fallback.
When you are done, send "DONE" and nothing else, and we will grade your submission.
"""
    + """
You may find it useful to search for particular keywords in the codebase using the `grep` command via subprocess:
```python
import subprocess
print(subprocess.check_output(['grep', '-irn', '"foo"', '/app/expensify']))
```
where `"foo"` is the keyword you want to search for.

Remember, you can go step by step. If you are unsure about a command, use a step to look up help e.g.
```python
import subprocess
print(subprocess.check_output(['grep', '--help']))
```

You should start by browsing the files in the codebase to understand the problem and formulate a plan.
"""
)
