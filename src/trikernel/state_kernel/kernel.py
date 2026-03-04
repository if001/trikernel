from langgraph.store.base import BaseStore
from trikernel.state_kernel.core.state_kernel_impl import StateKernel
from trikernel.state_kernel.protocols import StateKernelAPI


def create_state_kernel(memory_store: BaseStore) -> StateKernelAPI:
    return StateKernel(memory_store=memory_store)
