from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkSender(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...


@runtime_checkable
class WorkReceiver(Protocol):
    async def recv_json(self) -> dict[str, Any]: ...


@runtime_checkable
class ResultSender(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...


@runtime_checkable
class ResultReceiver(Protocol):
    async def recv_json(self) -> dict[str, Any]: ...


class ZmqWorkSender(WorkSender):
    def __init__(self, endpoint: str) -> None:
        self._socket = _create_socket(endpoint, bind=True, socket_type="PUSH")

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._socket.send_json(payload)


class ZmqWorkReceiver(WorkReceiver):
    def __init__(self, endpoint: str) -> None:
        self._socket = _create_socket(endpoint, bind=False, socket_type="PULL")

    async def recv_json(self) -> dict[str, Any]:
        return await self._socket.recv_json()


class ZmqResultSender(ResultSender):
    def __init__(self, endpoint: str) -> None:
        self._socket = _create_socket(endpoint, bind=False, socket_type="PUSH")

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._socket.send_json(payload)


class ZmqResultReceiver(ResultReceiver):
    def __init__(self, endpoint: str) -> None:
        self._socket = _create_socket(endpoint, bind=True, socket_type="PULL")

    async def recv_json(self) -> dict[str, Any]:
        return await self._socket.recv_json()


def _create_socket(endpoint: str, *, bind: bool, socket_type: str):
    try:
        import zmq  # type: ignore
        import zmq.asyncio  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyzmq is required for the composition layer") from exc
    context = zmq.asyncio.Context.instance()
    socket = context.socket(getattr(zmq, socket_type))
    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)
    return socket
