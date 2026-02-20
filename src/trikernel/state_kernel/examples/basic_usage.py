from trikernel.state_kernel.kernel import StateKernel


if __name__ == "__main__":
    state = StateKernel()
    task_id = state.task_create("user_request", {"user_message": "hello"})
    task = state.task_get(task_id)
    print(task)
