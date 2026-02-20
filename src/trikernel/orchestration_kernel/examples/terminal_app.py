import asyncio
from typing import Iterable

from ui.terminal import TerminalUI
from trikernel.orchestration_kernel.models import LLMResponse
from trikernel.orchestration_kernel.payloads import extract_llm_input
from trikernel.orchestration_kernel.runners import SingleTurnRunner
from trikernel.execution.session import TrikernelSession
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel


class EchoLLM:
    def generate(self, task, tools) -> LLMResponse:
        llm_input = extract_llm_input(task.payload or {})
        message = llm_input.get("message", "")
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
    session = TrikernelSession(state, tool_kernel, runner, llm)

    ui.write_output("Type a message. Empty input exits.")
    while True:
        user_input = ui.read_input("message> ").strip()
        if not user_input:
            break
        result = session.send_message(user_input, stream=False)
        ui.write_output(result.message or "")
        ui.write_stream(stream_chunks("(stream) done"))


if __name__ == "__main__":
    asyncio.run(main())
