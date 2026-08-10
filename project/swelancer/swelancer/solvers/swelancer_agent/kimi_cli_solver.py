"""Drive SWE-Lancer with Kimi Code CLI running *inside* the task container.

The other solvers in this package own the agent loop themselves: they hold the
conversation, hand the model a `python` tool, and execute whatever comes back
through alcatraz. That loop was written for Kimi K2.7 and shaped around what it
would emit.

This solver gives the loop away. Kimi Code CLI (MoonshotAI/kimi-code, MIT) is
Moonshot's own terminal agent and the current release targets K3, so it already
carries the tool schemas, context management and subagents that this repo's
solver approximates. Running it inside the task container means its file and
shell tools operate directly on `/app/expensify`, and grading is untouched --
the grader only ever looks at the repository state the agent leaves behind.

    SWELANCER_SOLVER=swelancer.solvers.swelancer_agent.kimi_cli_solver:KimiCliSolver

Inference stays on the host. `mlx_lm.server` is OpenAI-compatible and Kimi Code
reaches it at `host.docker.internal`, which is verified reachable from these
images (HTTP 200 against /v1/models); the CLI needs no protocol shim.

Two things worth knowing before changing this file:

- `max_context_size` is mandatory in a `[models.*]` block. Without it the CLI
  fails config validation at startup with "expected number, received undefined".
- The binary installs to `/root/.kimi-code/bin/kimi` -- note `kimi`, not
  `kimi-code` -- and the installer only adds it to PATH via `.bashrc`, which a
  non-interactive `send_shell_command` does not source. It is invoked by
  absolute path here for that reason.

Token accounting is left at zero: the CLI owns the conversation and does not
report usage back. `results.csv` will show 0 input/output tokens for runs made
with this solver, which is a reporting gap rather than a failed rollout -- read
correctness from `correct`/`earned`.
"""

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

KIMI_HOME = "/root/.kimi-code"
KIMI_BIN = f"{KIMI_HOME}/bin/kimi"
KIMI_CONFIG_PATH = f"{KIMI_HOME}/config.toml"

AGENT_PROMPT = """\
You are working in a checkout of the Expensify repository at /app/expensify, which \
is your working directory. Fix the issue described above by editing the repository \
directly with your tools. Do not ask the user questions -- you are running \
unattended, and there is nobody to answer. When you believe the fix is complete, \
stop; the repository state you leave behind is what gets graded."""


@chz.chz
class KimiCliSolver(PythonCodingSolver):
    name: str = chz.field(default="KimiCliSolver")
    computer_runtime: ComputerRuntime = chz.field(default_factory=AlcatrazComputerRuntime)
    runtime_config: RuntimeConfig = chz.field(default_factory=RuntimeConfig)

    model: str = chz.field(
        default="local-k3",
        doc="Model alias declared in the generated config.toml.",
    )
    model_id: str = chz.field(
        default="/Users/hello/mac_workspace/models/kimi-k3-reap73-mlx-mxfp4-q8",
        doc="Model id the OpenAI-compatible server answers to. mlx_lm.server "
        "identifies models by the path passed to --model, so this is a host path "
        "rather than a name; /v1/models on the server lists the accepted value.",
    )
    base_url: str = chz.field(
        default="http://host.docker.internal:8080/v1",
        doc="Inference endpoint as seen from inside the task container.",
    )
    max_context_size: int = chz.field(default=262144)
    install_url: str = chz.field(default="https://code.kimi.com/kimi-code/install.sh")
    rollout_timeout: int = chz.field(
        default=10800,
        doc="Wall-clock cap for one `kimi -p` invocation, in seconds. The CLI has "
        "no turn limit of its own here, so this is the only bound on a rollout.",
    )

    def shortname(self) -> str:
        return "kimi-cli"

    def _config_toml(self) -> str:
        return (
            f'default_model = "{self.model}"\n\n'
            "[providers.local-mlx]\n"
            'type = "openai"\n'
            f'base_url = "{self.base_url}"\n'
            # The CLI refuses to start with no credential, and the local server
            # does not check one; any non-empty string satisfies both.
            'api_key = "local"\n\n'
            f"[models.{self.model}]\n"
            'provider = "local-mlx"\n'
            f'model = "{self.model_id}"\n'
            f"max_context_size = {self.max_context_size}\n"
        )

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        assert isinstance(task, SWELancerTask), (
            f"KimiCliSolver only supports SWELancerTasks, got {type(task)}"
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

                ctx_logger.info("Installing Kimi Code CLI...", destinations=["run"])
                install = await computer.send_shell_command(
                    f"curl -fsSL {shlex.quote(self.install_url)} | bash"
                )
                if install.exit_code != 0:
                    raise RolloutSystemError(
                        f"Kimi Code CLI install failed ({install.exit_code}): "
                        f"{install.output.decode(errors='replace')[-2000:]}"
                    )

                await computer.check_shell_command(f"mkdir -p {shlex.quote(KIMI_HOME)}")
                await computer.upload(self._config_toml().encode(), KIMI_CONFIG_PATH)

                # `doctor` validates the config without contacting the model, so a
                # bad provider block fails here rather than as a mid-rollout stall.
                doctor = await computer.send_shell_command(f"{KIMI_BIN} doctor")
                ctx_logger.info(
                    f"kimi doctor: {doctor.output.decode(errors='replace')[-1000:]}",
                    destinations=["run"],
                )
                if doctor.exit_code != 0:
                    raise RolloutSystemError("Kimi Code CLI config failed validation")

                ctx_logger.info("Running Kimi Code CLI rollout...", destinations=["run"])
                # `-p` runs one prompt non-interactively and is deliberately used
                # bare: the CLI rejects `--auto` and `-y` alongside it ("Cannot
                # combine --prompt with --auto"), because prompt mode has no
                # approval step to auto-approve in the first place.
                cmd = (
                    f"cd /app/expensify && timeout {self.rollout_timeout} "
                    f"{KIMI_BIN} -p {shlex.quote(prompt)}"
                )
                result = await computer.send_shell_command(cmd)
                output = result.output.decode(errors="replace")
                ctx_logger.info(
                    f"Kimi CLI exited {result.exit_code}; tail: {output[-4000:]}",
                    destinations=["run"],
                )
                if result.exit_code == 124:
                    ctx_logger.info(
                        f"Kimi CLI hit the {self.rollout_timeout}s cap; grading current state.",
                        destinations=["run"],
                    )

                # Anything the agent left in /app/tests would let a later step see
                # the graded tests; the CLI never unzips them, but clear the path
                # the same way the other solvers do before handing over to grading.
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
