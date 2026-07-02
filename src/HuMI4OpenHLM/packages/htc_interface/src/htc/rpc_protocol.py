"""RPC protocol definitions for HTC recorder/server."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict


class RpcErrorCode(str, enum.Enum):
    """RPC error codes."""

    UNKNOWN_METHOD = "unknown_method"
    INVALID_PARAMS = "invalid_params"
    INTERNAL_ERROR = "internal_error"
    TIMEOUT = "timeout"


class RpcRequest(TypedDict):
    """Represents an RPC request message."""

    id: str
    method: str
    params: NotRequired[dict[str, Any]]


class RpcResponse(TypedDict):
    """Represents an RPC response message."""

    id: str
    result: Any


class RpcError(TypedDict):
    """Represents an RPC error in the response message."""

    code: str
    message: str


@dataclass(frozen=True)
class WireMessage:
    """
    Unified wire message for RPC communication.
    """

    payload: str
    """JSON-encoded RpcRequest or RpcResponse."""

    @classmethod
    def from_payload_obj(
        cls, obj: RpcRequest | RpcResponse | RpcError
    ) -> "WireMessage":
        """Create a WireMessage from a payload object."""
        return cls(payload=json.dumps(obj))

    def to_req(self) -> RpcRequest:
        """Decode the payload into an RpcRequest."""
        decoded = json.loads(self.payload)
        # Type check
        if "id" not in decoded or "method" not in decoded:
            raise ValueError("Payload is not a valid RpcRequest.")
        return decoded

    def to_resp_or_err(self) -> RpcResponse | RpcError:
        """Decode the payload into an RpcResponse or RpcError."""
        decoded = json.loads(self.payload)
        # Type check
        if "id" in decoded and "result" in decoded:
            return decoded  # RpcResponse
        elif "code" in decoded and "message" in decoded:
            return decoded  # RpcError
        else:
            raise ValueError(
                f"Payload is not a valid RpcResponse or RpcError: {decoded}"
            )


def make_request(
    id_: str, method: str, params: dict[str, Any] | None = None
) -> RpcRequest:
    req: RpcRequest = {"id": id_, "method": method}
    if params is not None:
        req["params"] = params
    return req


def make_response_ok(id_: str, result: Any) -> RpcResponse:
    return {"id": id_, "result": result}


def make_response_err(code: RpcErrorCode, message: str) -> RpcError:
    return {"code": code.value, "message": message}


# Delimiter for framing messages over ZMQ
# ROUTER receive: [client_id, b"", payload]
# ROUTER send:    [client_id, b"", payload]
# DEALER send/receive: [b"", payload]
FRAME_EMPTY_DELIM: bytes = b""
