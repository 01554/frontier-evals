"""Drive SWE-Lancer with Qwen Code CLI running *inside* the task container.

Same shape as kimi_cli_solver.py, with the maker's-own-agent principle applied
to Qwen models: Qwen Code (QwenLM/qwen-code, Apache-2.0, a Gemini-CLI fork) is
Alibaba's terminal agent, so a Qwen backend gets driven by Qwen's own tool
schemas and context management, the way K3 was driven by Kimi Code CLI.

Differences from the Kimi solver, all mechanical:
  - install: `npm install -g @qwen-code/qwen-code` (the Expensify container is
    a Node repo, so npm is already there) instead of an install.sh
  - config: OpenAI-compatible env vars (OPENAI_BASE_URL / OPENAI_API_KEY /
    OPENAI_MODEL) on the invocation, instead of a config.toml
  - rollout: `qwen -p <prompt> --yolo`; unlike `kimi -p`, the Gemini-CLI
    lineage allows auto-approval together with prompt mode, and needs it --
    without --yolo, prompt mode still gates shell/file tools.

Token accounting stays at zero, as with the Kimi solver: the CLI owns the
conversation and does not report usage. Read correctness from
`correct`/`earned`.
"""

import os
import shlex
from typing import Any, AsyncGenerator

import structlog
from typing_extensions import override

import chz
from nanoeval.contrib.utils import run_with_startup_timeout
from nanoeval.eval import RolloutSystemError
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import FinalResult, Step
from nanoeval.solvers.computer_tasks.task import ComputerTask
from swelancer.eval import SWELancerTask

from .solver import STARTUP_TIMEOUT, RuntimeConfig
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerRuntime
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerRuntime

logger = structlog.stdlib.get_logger(component=__name__)

AGENT_PROMPT = """\
You are working in a checkout of the Expensify repository at /app/expensify, which \
is your working directory. Fix the issue described above by editing the repository \
directly with your tools. Do not ask the user questions -- you are running \
unattended, and there is nobody to answer. When you believe the fix is complete, \
stop; the repository state you leave behind is what gets graded."""


@chz.chz
class QwenCliSolver(PythonCodingSolver):
    name: str = chz.field(default="QwenCliSolver")
    computer_runtime: ComputerRuntime = chz.field(default_factory=AlcatrazComputerRuntime)
    runtime_config: RuntimeConfig = chz.field(default_factory=RuntimeConfig)

    model_id: str = chz.field(
        default="qwen3.8",
        doc="Model name sent to the OpenAI-compatible server (llama-server "
        "accepts any string).",
    )
    base_url: str = chz.field(
        default="http://host.docker.internal:8090/v1",
        doc="Inference endpoint as seen from inside the task container.",
    )
    npm_package: str = chz.field(default="@qwen-code/qwen-code@0.21.11")
    rollout_timeout: int = chz.field(
        default=10800,
        doc="Wall-clock cap for one `qwen -p` invocation, in seconds.",
    )

    def shortname(self) -> str:
        return "qwen-cli"

    # node lives in an nvm dir that is NOT on PATH in this container's
    # non-interactive shells (verified: only ~/.nvm/versions/node/v20.18.0
    # exists; `bash -lc which node` fails). Resolve it by glob so a rebuilt
    # image with a newer node keeps working.
    PATH_PREFIX = 'export PATH="$(ls -d $HOME/.nvm/versions/node/*/bin | head -1):$PATH" && '

    def _env_prefix(self) -> str:
        # Qwen Code's OpenAI-compatible mode is configured entirely by env.
        # The local server checks no credential; any non-empty key satisfies
        # the CLI's presence check.
        return (
            f"OPENAI_BASE_URL={shlex.quote(self.base_url)} "
            "OPENAI_API_KEY=local "
            f"OPENAI_MODEL={shlex.quote(self.model_id)} "
            # A slow local server legitimately spends 400+ s in prefill before
            # the first streamed chunk; qwen-code's default 240 s idle timeout
            # kills the request first ("No stream activity for 240000ms after
            # 0 chunks"). 0 disables it; the rollout wall-cap still bounds us.
            "QWEN_STREAM_IDLE_TIMEOUT_MS=0 "
            "QWEN_CODE_SUPPRESS_YOLO_WARNING=1 "
        )

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        assert isinstance(task, SWELancerTask), (
            f"QwenCliSolver only supports SWELancerTasks, got {type(task)}"
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
                assert "content" in task.prompt[0]
                assert isinstance(task.prompt[0]["content"], str)
                prompt = task.prompt[0]["content"] + "\n\n" + AGENT_PROMPT
                # Optional counter-note against the benchmark prompt's phantom
                # "reply in ```python blocks" scaffold instruction (that
                # scaffold does not exist in CLI rollouts). Same idea as the
                # rtx-side promptv1; text version-controlled here. Enable by
                # exporting QWEN_PROMPT_NOTE=v1m; label the run accordingly.
                if os.environ.get("QWEN_PROMPT_NOTE") == "v1m":
                    prompt += (
                        "\n\nIMPORTANT correction to the instructions above: "
                        "in this environment, ```python blocks in your replies "
                        "are NOT executed by anyone, and <user-tool> does not "
                        "exist. The ONLY way to act is your own tool calls "
                        "(shell, file read/write/edit). Never wait for an "
                        "external executor."
                    )

                ctx_logger.info("Installing Qwen Code CLI...", destinations=["run"])
                install = await computer.send_shell_command(
                    f"{self.PATH_PREFIX}npm install -g {shlex.quote(self.npm_package)} 2>&1 | tail -5"
                )
                if install.exit_code != 0:
                    raise RolloutSystemError(
                        f"Qwen Code CLI install failed ({install.exit_code}): "
                        f"{install.output.decode(errors='replace')[-2000:]}"
                    )

                version = await computer.send_shell_command(f"{self.PATH_PREFIX}qwen --version")
                ctx_logger.info(
                    f"qwen --version: {version.output.decode(errors='replace')[-500:]}",
                    destinations=["run"],
                )
                if version.exit_code != 0:
                    raise RolloutSystemError("Qwen Code CLI not runnable after install")

                ctx_logger.info("Running Qwen Code CLI rollout...", destinations=["run"])
                # positional prompt (the -p flag is deprecated in qwen-code
                # 0.15); -y/--yolo is required or prompt mode still gates tools.
                cmd = (
                    f"{self.PATH_PREFIX}cd /app/expensify && {self._env_prefix()}"
                    f"timeout {self.rollout_timeout} "
                    f"qwen -y {shlex.quote(prompt)}"
                )
                result = await computer.send_shell_command(cmd)
                output = result.output.decode(errors="replace")
                ctx_logger.info(
                    f"Qwen CLI exited {result.exit_code}; tail: {output[-4000:]}",
                    destinations=["run"],
                )
                if result.exit_code == 124:
                    ctx_logger.info(
                        f"Qwen CLI hit the {self.rollout_timeout}s cap; grading current state.",
                        destinations=["run"],
                    )

                await computer.send_shell_command("rm -rf /app/tests")

                try:
                    grade = await task.grade(computer, self.runtime_config)
                except Exception as e:
                    raise RolloutSystemError(f"Error during grading: {e}") from e

                yield FinalResult(grade=grade)
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
