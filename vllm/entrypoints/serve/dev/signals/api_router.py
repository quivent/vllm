# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime control for inference-signal capture.

Which hooks exist, and whether the model runs eager, are fixed when the engine
loads. What those hooks *record* is not: this router moves the active tier up
and down inside the ceiling set by `--signal-capture-max-tier`, so a server can
sit at `off` for normal traffic and be switched to recording for a window
without a restart.
"""

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

_NOT_CONFIGURED = (
    "Signal capture/injection is not configured on this engine. Restart with "
    "--signal-capture-max-tier (and --signal-capture-dir), or --signal-inject-from."
)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _first(results) -> dict | None:
    """Capture runs on TP rank 0 only, so take the first non-null result."""
    if not results:
        return None
    for result in results:
        if result is not None:
            return result
    return None


@router.get("/signals/status")
async def signals_status(raw_request: Request):
    """Report the active tier, the ceiling, and what is being tapped."""
    results = await engine_client(raw_request).collective_rpc(
        method="get_signal_capture_status"
    )
    status = _first(results)
    if status is None:
        return JSONResponse(
            content={
                "enabled": False,
                "detail": "Signal capture is not configured on this engine. "
                "Restart with --signal-capture-max-tier and --signal-capture-dir.",
            }
        )
    return JSONResponse(content=status)


@router.post("/signals/control")
async def signals_control(raw_request: Request):
    """Change the active tier, token reduction, or layer selection.

    Body (all fields optional)::

        {"tier": "residual_raw", "tokens": "last"}

    `tier` may be any tier at or below the startup ceiling; `off` stops
    recording while leaving the hooks in place. Returns the resulting status.
    """
    body = await raw_request.json() if await raw_request.body() else {}
    kwargs = {key: body[key] for key in ("tier", "tokens", "layers") if body.get(key)}
    if not kwargs:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Provide at least one of: tier, tokens, layers.",
        )

    logger.info("Signal capture control: %s", kwargs)
    try:
        results = await engine_client(raw_request).collective_rpc(
            method="set_signal_capture", kwargs=kwargs
        )
    except Exception as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)
        ) from exc

    status = _first(results)
    if status is None:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail="Signal capture is not configured on this engine. Restart "
            "with --signal-capture-max-tier and --signal-capture-dir.",
        )
    return JSONResponse(content=status)


@router.get("/signals/injection")
async def signals_injection_status(raw_request: Request):
    """Report the residual injection currently installed, if any."""
    results = await engine_client(raw_request).collective_rpc(
        method="get_signal_injection_status"
    )
    status = _first(results)
    if status is None:
        return JSONResponse(content={"enabled": False, "detail": _NOT_CONFIGURED})
    return JSONResponse(content=status)


@router.post("/signals/inject")
async def signals_inject(raw_request: Request):
    """Inject a captured residual back into the forward pass.

    Body::

        {
            "source": "/path/turn.safetensors",  # omit to clear the injection
            "layer": 63,  # default: where it was recorded
            "alpha": 1.0,
            "mode": "add" | "replace",
            "positions": "first" | "all",
            "row": -1,
        }

    `add` steers along the captured direction; `replace` makes the captured
    state *be* the state at that layer. `first` seeds each request once --
    the "starting point" reading -- while `all` holds it for the whole turn.
    """
    body = await raw_request.json() if await raw_request.body() else {}
    kwargs = {
        key: body[key]
        for key in ("source", "layer", "alpha", "mode", "positions", "signal", "row")
        if key in body
    }

    logger.info("Signal injection: %s", kwargs or "clear")
    try:
        results = await engine_client(raw_request).collective_rpc(
            method="set_signal_injection", kwargs=kwargs
        )
    except Exception as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)
        ) from exc

    status = _first(results)
    if status is None:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value, detail=_NOT_CONFIGURED
        )
    return JSONResponse(content=status)


def attach_router(app: FastAPI):
    app.include_router(router)
