from trikernel.composition.ui import TerminalUI
from trikernel.orchestration_kernel import OllamaLLM, RunnerContext, SingleTurnRunner
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel


def main() -> None:
    ui = TerminalUI()
    state = StateKernel()
    tool_kernel = ToolKernel()
    runner = SingleTurnRunner()
    llm = OllamaLLM()

    ui.write_output("Type a message for Ollama. Empty input exits.")
    while True:
        text = ui.read_input("message> ").strip()
        if not text:
            break
        task_id = state.task_create("user_request", {"message": text})
        task = state.task_get(task_id)
        context = RunnerContext(
            runner_id="main",
            state_api=state,
            tool_api=tool_kernel,
            llm_api=llm,
            tool_llm_api=ToolOllamaLLM(),
            stream=True,
        )
        result = runner.run(task, context)
        if result.stream_chunks:
            ui.write_stream(result.stream_chunks)
        else:
            ui.write_output(result.user_output or "")


if __name__ == "__main__":
    main()
