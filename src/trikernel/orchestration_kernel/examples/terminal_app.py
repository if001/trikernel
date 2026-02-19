import asyncio
from typing import Iterable, List

from trikernel.composition.ui import TerminalUI
from trikernel.orchestration_kernel.models import LLMResponse, LLMToolCall
from trikernel.orchestration_kernel.runners import SingleTurnRunner
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.orchestration_kernel.models import RunnerContext


class EchoLLM:
    def generate(self, task, tools) -> LLMResponse:
        message = task.payload.get("message", "")
        return LLMResponse(user_output=f"Echo: {message}", tool_calls=[])


def stream_chunks(text: str, size: int = 6) -> Iterable[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


async def main() -> None:
    ui = TerminalUI()
    state = StateKernel()
    tool_kernel = ToolKernel()
    runner = SingleTurnRunner()
    llm = EchoLLM()

    ui.write_output("Type a message. Empty input exits.")
    while True:
        user_input = ui.read_input("message> ").strip()
        if not user_input:
            break
        task_id = state.task_create("user_request", {"message": user_input})
        task = state.task_get(task_id)
        context = RunnerContext(
            runner_id="main",
            state_api=state,
            tool_api=tool_kernel,
            llm_api=llm,
        )
        result = runner.run(task, context)
        ui.write_output(result.user_output or "")
        ui.write_stream(stream_chunks("(stream) done"))


if __name__ == "__main__":
    asyncio.run(main())
