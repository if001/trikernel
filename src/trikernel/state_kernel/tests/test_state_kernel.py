from trikernel.state_kernel.kernel import StateKernel


def test_task_lifecycle(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    task_id = state.task_create("user_request", {"message": "hi"})
    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "queued"

    claimed = state.task_claim({"task_type": "user_request"}, "runner", 5)
    assert claimed == task_id
    task = state.task_get(task_id)
    assert task.state == "running"

    state.task_complete(task_id)
    task = state.task_get(task_id)
    assert task.state == "done"
