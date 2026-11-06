"""Cursor ACP transport over the stock Hermes ACP client."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

from agent.acp_client import (
    ACPClient,
    _effective_timeout,
    _format_messages_as_prompt,
    _jsonrpc_error,
    _jsonrpc_result,
    _model_config_option,
    _model_selection_request,
    _session_model_ids,
)
from agent.acp_openai_bridge import completion_to_stream_chunks, extract_tool_calls_from_text

from .fence import Fence
from .model_pin import ModelPinError, resolve_model_pin
from .normalize import normalize_tool_call_text

_INITIALIZE_PARAMS = {
    "protocolVersion": 1,
    "clientCapabilities": {
        "fs": {"readTextFile": False, "writeTextFile": False},
        "terminal": False,
        "_meta": {"parameterizedModelPicker": True},
    },
    "clientInfo": {"name": "hermes-agent", "title": "Hermes Agent", "version": "0.0.0"},
}
_PIN_FAILURES = (RuntimeError, TimeoutError, OSError, TypeError, AttributeError)


class CursorACPClient(ACPClient):
    """One short ACP session per completion, inside a private fence."""

    default_command = "agent"
    default_args = ("acp",)
    marker_base_url = "acp://cursor"
    placeholder_model = "cursor-acp-unset"
    label = "Cursor ACP"
    SUPPORTS_HERMES_TOOL_CALLS = False

    def __init__(self, **kwargs: Any) -> None:
        self._fence = Fence.create()
        self._fence_closed = False
        self._request_process: subprocess.Popen[str] | None = None
        self._live_sessions = 0
        self._live_sessions_lock = threading.Lock()
        kwargs["acp_cwd"] = str(self._fence.cwd)
        super().__init__(**kwargs)

    def close(self) -> None:
        self._fence_closed = True
        super().close()
        self._fence.close()

    def cancel(self) -> None:
        """Terminate the child started for the current request. No session/cancel."""
        proc = self._request_process
        if proc is not None:
            self._terminate_process(proc)

    def _spawn(self) -> subprocess.Popen[str]:
        argv = self._fence.command_argv()
        env = self._fence.child_env(os.environ)
        from hermes_cli._subprocess_compat import windows_hide_flags

        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._acp_cwd,
                env=env,
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            command = argv[0] if argv else self.default_command
            raise RuntimeError(
                f"Could not start {self.label} command '{command}'. {self._missing_command_hint()}"
            ) from exc
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError(f"{self.label} process did not expose stdin/stdout pipes.")
        with self._active_process_lock:
            self._active_processes.add(proc)
            self.is_closed = False
        self._request_process = proc
        return proc

    @contextlib.contextmanager
    def _live_turn(self, session_holder: dict[str, str | None]) -> Iterator[None]:
        """Count this prompt until its cleanup, then drop the shared tmpdir last."""
        with self._live_sessions_lock:
            self._live_sessions += 1
        try:
            yield
        finally:
            with self._live_sessions_lock:
                try:
                    session_id = session_holder.get("id")
                    if session_id and not self._fence_closed:
                        self._fence.cleanup_turn(
                            session_id,
                            delete_tmpdir=self._live_sessions == 1,
                        )
                finally:
                    self._live_sessions -= 1

    @contextlib.contextmanager
    def _session(
        self,
        timeout_seconds: float,
        *,
        allow_file_requests: bool = True,
        session_holder: dict[str, str | None] | None = None,
    ) -> Iterator[tuple[dict[str, Any], Callable[..., Any]]]:
        if self._fence_closed:
            raise RuntimeError("Cursor ACP client is closed.")
        self._fence.prepare_spawn()
        proc = self._spawn()
        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _decode(line: str) -> dict[str, Any]:
            try:
                return json.loads(line)
            except Exception:
                return {"raw": line.rstrip("\n")}

        def _pump(stream, sink) -> None:
            for line in stream or ():
                sink(line)

        threading.Thread(
            target=_pump, args=(proc.stdout, lambda line: inbox.put(_decode(line))), daemon=True
        ).start()
        stderr_pump = threading.Thread(
            target=_pump,
            args=(proc.stderr, lambda line: stderr_tail.append(line.rstrip("\n"))),
            daemon=True,
        )
        stderr_pump.start()
        request_ids = iter(range(1, 1 << 62))
        session_deadline = time.monotonic() + timeout_seconds

        def _request(
            method: str,
            params: dict[str, Any],
            *,
            text_parts: list[str] | None = None,
            reasoning_parts: list[str] | None = None,
        ) -> Any:
            request_id = next(request_ids)
            try:
                proc.stdin.write(
                    json.dumps(
                        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
                    )
                    + "\n"
                )
                proc.stdin.flush()
            except OSError as exc:
                if not isinstance(exc, BrokenPipeError) and exc.errno != errno.EINVAL:
                    raise
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
                if proc.poll() is None:
                    raise
            deadline = session_deadline
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                    allow_file_requests=allow_file_requests,
                ) or msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    raise RuntimeError(self._request_error(method, msg.get("error") or {}))
                return msg.get("result")
            if proc.poll() is not None:
                stderr_pump.join(timeout=1.0)
                stderr_text = "\n".join(stderr_tail).strip()
                if (vendor_error := self.early_exit_error(stderr_text)) is not None:
                    raise RuntimeError(vendor_error)
                raise RuntimeError(
                    f"{self.label} process exited early: {stderr_text or f'exit code {proc.returncode}'}"
                )
            raise TimeoutError(f"Timed out waiting for {self.label} response to {method}.")

        try:
            _request("initialize", _INITIALIZE_PARAMS)
            session = _request("session/new", {"cwd": self._acp_cwd, "mcpServers": []}) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError(f"{self.label} did not return a sessionId.")
            if session_holder is not None:
                session_holder["id"] = session_id
            _request("session/set_mode", {"sessionId": session_id, "modeId": "ask"})
            yield session, _request
        finally:
            self._release_process(proc)

    def list_models(self, *, timeout_seconds: float = 15.0) -> list[str]:
        """One short session, then delete that session's acp-sessions directory."""
        session_holder: dict[str, str | None] = {"id": None}
        with self._live_turn(session_holder):
            with self._session(
                timeout_seconds, allow_file_requests=False, session_holder=session_holder
            ) as (session, _):
                return _session_model_ids(session, self.model_entry_disabled)

    def _run_prompt(
        self, prompt_text: str, *, timeout_seconds: float, model: str | None = None
    ) -> tuple[str, str]:
        session_holder: dict[str, str | None] = {"id": None}
        with self._live_turn(session_holder):
            with self._session(timeout_seconds, session_holder=session_holder) as (session, _request):
                session_id = str(session.get("sessionId") or "").strip()
                available_ids = _session_model_ids(session, self.model_entry_disabled)
                requested = "" if model is None else str(model)
                pinned = resolve_model_pin(
                    requested=requested,
                    available_ids=available_ids,
                    placeholder_model=self.placeholder_model,
                    model_entry_disabled=self._requested_model_disabled(session, requested),
                )
                try:
                    selection = _model_selection_request(
                        session,
                        pinned,
                        placeholder=self.placeholder_model,
                        disabled=self.model_entry_disabled,
                    )
                    if selection is None:
                        raise ModelPinError(available_ids, f"model pin for {pinned!r} was not sent")
                    _request(*selection)
                except ModelPinError:
                    raise
                except _PIN_FAILURES as exc:
                    raise ModelPinError(available_ids, f"model pin failed: {exc}") from exc
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                _request(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": prompt_text}],
                    },
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                )
                return "".join(text_parts), "".join(reasoning_parts)

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [], model=model, tools=tools, tool_choice=tool_choice
        )
        response_text, reasoning = self._run_prompt(
            prompt_text, timeout_seconds=_effective_timeout(timeout), model=model
        )
        tool_calls, cleaned_text = extract_tool_calls_from_text(normalize_tool_call_text(response_text))
        message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning or None,
            reasoning_content=reasoning or None,
            reasoning_details=None,
        )
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message, finish_reason="tool_calls" if tool_calls else "stop"
                )
            ],
            usage=None,
            model=model or self.placeholder_model,
        )
        return completion_to_stream_chunks(completion) if stream else completion

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
        allow_file_requests: bool = True,
    ) -> bool:
        method = msg.get("method")
        if method in {"fs/read_text_file", "fs/write_text_file"} or (
            isinstance(method, str) and method.startswith("terminal/")
        ):
            return self._answer(
                process,
                _jsonrpc_error(
                    msg.get("id"),
                    -32601,
                    f"ACP client method '{method}' is not supported by Hermes yet.",
                ),
            )
        if method == "session/request_permission":
            return self._answer(
                process,
                _jsonrpc_result(msg.get("id"), {"outcome": {"outcome": "reject-once"}}),
            )
        return super()._handle_server_message(
            msg,
            process=process,
            cwd=cwd,
            text_parts=text_parts,
            reasoning_parts=reasoning_parts,
            allow_file_requests=allow_file_requests,
        )

    def _answer(self, process: subprocess.Popen[str], response: dict[str, Any]) -> bool:
        if process.stdin is None:
            return True
        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True

    def _requested_model_disabled(self, session: dict[str, Any], requested: str) -> bool:
        chosen = requested.strip()
        if not chosen:
            return False
        option = _model_config_option(session)
        if option is not None:
            entries = option.get("options") or []
            key = "value"
        else:
            entries = (session.get("models") or {}).get("availableModels") or []
            key = "modelId"
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get(key) or "").strip() == chosen and self.model_entry_disabled(entry):
                return True
        return False
