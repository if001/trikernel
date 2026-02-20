from ui.terminal import TerminalUI
from trikernel.orchestration_kernel import OllamaLLM, SingleTurnRunner
from trikernel.execution.session import TrikernelSession
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel


def main() -> None:
    ui = TerminalUI()
    state = StateKernel()
    tool_kernel = ToolKernel()
    runner = SingleTurnRunner()
    llm = OllamaLLM()
    session = TrikernelSession(state, tool_kernel, runner, llm, ToolOllamaLLM())

    ui.write_output("Type a message for Ollama. Empty input exits.")
    while True:
        text = ui.read_input("message> ").strip()
        if not text:
            break
        result = session.send_message(text, stream=True)
        if result.stream_chunks:
            ui.write_stream(result.stream_chunks)
        else:
            ui.write_output(result.message or "")


if __name__ == "__main__":
    main()
