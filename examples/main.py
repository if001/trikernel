from __future__ import annotations

from ui.terminal import TerminalUI
from trikernel.orchestration_kernel import (
    OllamaLLM,
    SingleTurnRunner,
    PDCARunner,
    ToolLoopRunner,
)
from trikernel.api.session import TrikernelSession
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.tool_kernel.registry import register_default_tools


def main() -> None:
    ui = TerminalUI()
    state = StateKernel()
    tool_kernel = ToolKernel()
    register_default_tools(tool_kernel)

    # runner = SingleTurnRunner()
    # runner = PDCARunner()
    runner = ToolLoopRunner()
    llm = OllamaLLM()
    tool_llm = ToolOllamaLLM()
    session = TrikernelSession(state, tool_kernel, runner, llm, tool_llm)

    session.start_workers()
    ui.write_output("Type a message. Empty input exits.")
    try:
        while True:
            for message in session.drain_notifications():
                ui.write_output(message)
            user_input = ui.read_input("message> ").strip()
            if not user_input:
                break

            result = session.send_message(user_input, stream=False)
            if result.stream_chunks:
                ui.write_stream(result.stream_chunks)
            else:
                ui.write_output(result.message or "")
    finally:
        session.stop_workers()


if __name__ == "__main__":
    main()
