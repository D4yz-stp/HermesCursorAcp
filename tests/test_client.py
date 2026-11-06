"""Cursor ACP client: fence spawn, ask-mode session, visible pin, no native tools."""

import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

HERMES_ROOT = "/home/d4yz_/.hermes/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

from agent.acp_client import ACPClient
from cursor_acp.client import CursorACPClient

SESSION_ID = "sess-hca-1"
API_KEY = "provider-key-hca-should-not-leak"
FLAT_CALL = (
    '<tool_call>{"name":"canary_tool","arguments":{"path":"canary.txt"}}</tool_call>'
)
PREAMBLE = "You are being used as the active ACP agent backend for Hermes."


def rpc_result(msg, result):
    return {"jsonrpc": "2.0", "id": msg["id"], "result": result}


def rpc_error(msg, code, message):
    return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": code, "message": message}}


def server_request(method, params, message_id):
    return {"jsonrpc": "2.0", "id": message_id, "method": method, "params": params}


def text_chunk(kind, text):
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "update": {
                "sessionUpdate": kind,
                "content": {"type": "text", "text": text},
            }
        },
    }


class BlockingLines:
    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()

    def push(self, line: str) -> None:
        self._queue.put(line)

    def close(self) -> None:
        self._queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item


class FakeProcess:
    """Stdio stand-in. It never execs the agent binary."""

    def __init__(self, on_request) -> None:
        self.on_request = on_request
        self.stdout = BlockingLines()
        self.stderr = BlockingLines()
        self.written: list[str] = []
        self.terminated = False
        self.killed = False
        self.returncode = None
        self.saw_cli = False
        self.stdin = self

    def write(self, data: str) -> int:
        self.written.append(data)
        for line in data.splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            if "method" not in msg:
                continue
            for outbound in self.on_request(msg):
                self.stdout.push(json.dumps(outbound) + "\n")
        return len(data)

    def flush(self) -> None:
        return None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.stdout.close()
        self.stderr.close()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("agent", timeout if timeout is not None else 2)
        return self.returncode


def parsed_messages(proc: FakeProcess) -> list[dict]:
    messages = []
    for blob in proc.written:
        for line in blob.splitlines():
            if line.strip():
                messages.append(json.loads(line))
    return messages


def method_calls(proc: FakeProcess) -> list[dict]:
    return [msg for msg in parsed_messages(proc) if "method" in msg]


def result_messages(proc: FakeProcess) -> list[dict]:
    return [
        msg
        for msg in parsed_messages(proc)
        if "method" not in msg and ("result" in msg or "error" in msg)
    ]


class CursorACPClientTests(unittest.TestCase):
    def setUp(self) -> None:
        with mock.patch(
            "cursor_acp.client.subprocess.Popen",
            side_effect=AssertionError("constructor started a process"),
        ):
            self.client = CursorACPClient(
                api_key=API_KEY,
                command="/no/such/cursor-agent-hca",
                args=["acp", "--yolo", "--model", "auto"],
                not_a_cursor_kw="kept",
            )
        self.addCleanup(self._retire, self.client)

    def _retire(self, client) -> None:
        fence = getattr(client, "_fence", None)
        try:
            client.close()
        finally:
            if fence is not None and not fence._closed:
                fence.close()

    def _models_payload(self, available: tuple[str, ...]) -> dict:
        return {
            "sessionId": SESSION_ID,
            "models": {
                "currentModelId": available[0] if available else "",
                "availableModels": [{"modelId": model_id, "name": model_id} for model_id in available],
            },
        }

    def _plant_residue(self) -> tuple[Path, Path, Path]:
        session_dir = self.client._fence.config_dir / "acp-sessions" / SESSION_ID
        session_dir.mkdir(parents=True, exist_ok=True)
        session_canary = session_dir / "residue.txt"
        session_canary.write_text("session-residue")
        tmp_canary = self.client._fence.tmpdir / "residue.txt"
        tmp_canary.write_text("tmp-residue")
        outside = Path(tempfile.mkdtemp(prefix="cursor-acp-outside-")) / "kept.txt"
        outside.write_text("outside-residue")
        self.addCleanup(lambda: outside.parent.exists() and __import__("shutil").rmtree(outside.parent))
        return session_canary, tmp_canary, outside

    def _drive(
        self,
        *,
        model: str,
        messages: list | None = None,
        available: tuple[str, ...] = ("default", "grok-4.7"),
        session_result: dict | None = None,
        fail_method: str | None = None,
        on_method=None,
        on_prompt=None,
        tools=None,
        tool_choice=None,
        stream: bool = False,
        client=None,
    ):
        owner = client or self.client
        procs: list[FakeProcess] = []

        def handle(msg: dict) -> list[dict]:
            method = msg["method"]
            if on_method is not None:
                on_method(method)
            if fail_method == method:
                return [rpc_error(msg, -32603, f"{method} refused")]
            if method == "initialize":
                return [rpc_result(msg, {"protocolVersion": 1})]
            if method == "session/new":
                payload = session_result if session_result is not None else self._models_payload(available)
                return [rpc_result(msg, payload)]
            if method in {"session/set_mode", "session/set_model", "session/set_config_option"}:
                return [rpc_result(msg, {})]
            if method == "session/prompt":
                outbound = list(on_prompt() if on_prompt else [text_chunk("agent_message_chunk", "hello")])
                outbound.append(rpc_result(msg, {"stopReason": "end_turn"}))
                return outbound
            return [rpc_error(msg, -32601, f"unexpected {method}")]

        def popen(argv, **kwargs):
            proc = FakeProcess(handle)
            cli = Path(kwargs["cwd"]) / ".cursor" / "cli.json"
            proc.saw_cli = cli.is_file()
            proc.argv = list(argv)
            proc.kwargs = kwargs
            procs.append(proc)
            return proc

        with mock.patch("cursor_acp.client.subprocess.Popen", side_effect=popen):
            completion = owner.chat.completions.create(
                model=model,
                messages=messages
                or [{"role": "user", "content": "ping-sentinel"}],
                tools=tools,
                tool_choice=tool_choice,
                timeout=3,
                stream=stream,
            )
        self.assertEqual(len(procs), 1)
        return completion, procs[0]

    def test_constructor_builds_private_cwd_and_does_not_spawn(self) -> None:
        self.assertIsInstance(self.client, ACPClient)
        self.assertEqual(self.client.base_url, "acp://cursor")
        self.assertEqual(self.client._acp_cwd, str(self.client._fence.cwd))
        self.assertTrue(Path(self.client._acp_cwd).is_absolute())
        self.assertEqual(self.client._active_processes, set())
        with mock.patch(
            "cursor_acp.client.subprocess.Popen",
            side_effect=AssertionError("constructor started a process"),
        ):
            other = CursorACPClient(acp_cwd="/tmp", unknown_cursor_kw=True)
        self.addCleanup(self._retire, other)
        self.assertEqual(other._acp_cwd, str(other._fence.cwd))
        self.assertNotEqual(Path(other._acp_cwd), Path("/tmp"))

    def test_declared_transport_defaults(self) -> None:
        self.assertEqual(CursorACPClient.default_command, "agent")
        self.assertEqual(CursorACPClient.default_args, ("acp",))
        self.assertEqual(CursorACPClient.marker_base_url, "acp://cursor")
        self.assertEqual(CursorACPClient.placeholder_model, "cursor-acp-unset")
        self.assertEqual(CursorACPClient.label, "Cursor ACP")
        self.assertIs(CursorACPClient.SUPPORTS_HERMES_TOOL_CALLS, False)
        self.assertNotIn("supports_streaming", CursorACPClient.__dict__)
        self.assertNotIn("supports_vision", CursorACPClient.__dict__)

    def test_spawn_uses_agent_acp_and_fence_env_without_touching_the_parent(self) -> None:
        sentinel = "/tmp/hca-xdg-sentinel-not-a-directory"
        previous = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = sentinel
        try:
            before = os.environ.copy()
            captured = {}

            def popen(argv, **kwargs):
                captured["argv"] = list(argv)
                captured["kwargs"] = kwargs
                raise FileNotFoundError("stop before a real process")

            with mock.patch("cursor_acp.client.subprocess.Popen", side_effect=popen):
                with self.assertRaises(RuntimeError):
                    self.client._spawn()
            self.assertEqual(os.environ.copy(), before)
        finally:
            if previous is None:
                os.environ.pop("XDG_CONFIG_HOME", None)
            else:
                os.environ["XDG_CONFIG_HOME"] = previous

        self.assertEqual(captured["argv"], ["agent", "acp"])
        self.assertNotIn("--yolo", captured["argv"])
        self.assertNotIn("--model", captured["argv"])
        self.assertEqual(captured["kwargs"]["cwd"], str(self.client._fence.cwd))
        env = captured["kwargs"]["env"]
        self.assertEqual(env["XDG_CONFIG_HOME"], sentinel)
        self.assertEqual(env["HOME"], str(self.client._fence.home))
        self.assertEqual(env["CURSOR_CONFIG_DIR"], str(self.client._fence.config_dir))
        self.assertEqual(env["CURSOR_DATA_DIR"], str(self.client._fence.data_dir))
        self.assertEqual(env["TMPDIR"], str(self.client._fence.tmpdir))
        self.assertEqual(env["CURSOR_AGENT_DISABLE_DEBUG_LOG"], "1")
        self.assertNotEqual(env.get("CURSOR_API_KEY"), API_KEY)
        self.assertNotEqual(env.get("CURSOR_AUTH_TOKEN"), API_KEY)
        self.assertEqual("CURSOR_API_KEY" in env, "CURSOR_API_KEY" in before)
        self.assertEqual("CURSOR_AUTH_TOKEN" in env, "CURSOR_AUTH_TOKEN" in before)

    def test_missing_binary_names_cursor_acp(self) -> None:
        with mock.patch(
            "cursor_acp.client.subprocess.Popen",
            side_effect=FileNotFoundError("agent"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.client._spawn()
        self.assertIn("Cursor ACP", str(caught.exception))
        self.assertIn("agent", str(caught.exception))

    def test_session_initialize_disables_fs_and_terminal_and_sets_ask_mode(self) -> None:
        _completion, proc = self._drive(model="grok-4.7")
        calls = method_calls(proc)
        methods = [call["method"] for call in calls]
        self.assertEqual(
            methods,
            ["initialize", "session/new", "session/set_mode", "session/set_model", "session/prompt"],
        )
        self.assertNotIn("authenticate", methods)
        self.assertNotIn("session/cancel", methods)
        self.assertNotIn("subagents", json.dumps(calls[0]["params"]))
        capabilities = calls[0]["params"]["clientCapabilities"]
        self.assertEqual(calls[0]["params"]["protocolVersion"], 1)
        self.assertEqual(
            capabilities["fs"],
            {"readTextFile": False, "writeTextFile": False},
        )
        self.assertIs(capabilities["terminal"], False)
        self.assertEqual(capabilities["_meta"], {"parameterizedModelPicker": True})
        self.assertEqual(set(capabilities), {"fs", "terminal", "_meta"})
        self.assertEqual(calls[1]["params"]["mcpServers"], [])
        self.assertEqual(calls[1]["params"]["cwd"], str(self.client._fence.cwd))
        self.assertTrue(Path(calls[1]["params"]["cwd"]).is_absolute())
        self.assertEqual(calls[2]["params"], {"sessionId": SESSION_ID, "modeId": "ask"})
        self.assertTrue(proc.saw_cli)
        self.assertEqual(proc.argv, ["agent", "acp"])

    def test_prompt_is_a_single_stock_text_block(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ping-sentinel"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "canary_tool",
                    "description": "look",
                    "parameters": {"type": "object"},
                },
            }
        ]
        _completion, proc = self._drive(
            model="grok-4.7",
            messages=messages,
            tools=tools,
            tool_choice="none",
        )
        prompt = method_calls(proc)[-1]["params"]["prompt"]
        self.assertEqual([block["type"] for block in prompt], ["text"])
        text = prompt[0]["text"]
        self.assertIn(PREAMBLE, text)
        self.assertIn("ping-sentinel", text)
        self.assertIn("Tool choice hint:", text)
        self.assertIn("canary_tool", text)
        self.assertNotIn("grok-4.7", text)
        self.assertNotIn("image/png", text)
        self.assertNotIn("AAAA", text)

    def test_advertised_model_is_pinned_once(self) -> None:
        _completion, proc = self._drive(model="grok-4.7")
        pins = [call for call in method_calls(proc) if call["method"] == "session/set_model"]
        self.assertEqual(len(pins), 1)
        self.assertEqual(pins[0]["params"], {"sessionId": SESSION_ID, "modelId": "grok-4.7"})

    def test_advertised_default_is_pinned(self) -> None:
        _completion, proc = self._drive(model="default")
        pins = [call for call in method_calls(proc) if call["method"] == "session/set_model"]
        self.assertEqual(len(pins), 1)
        self.assertEqual(pins[0]["params"]["modelId"], "default")

    def test_config_option_pin_uses_the_base_selection_request(self) -> None:
        session = {
            "sessionId": SESSION_ID,
            "configOptions": [
                {
                    "id": "model",
                    "category": "model",
                    "options": [{"value": "default"}, {"value": "grok-4.7"}],
                }
            ],
        }
        _completion, proc = self._drive(model="grok-4.7", session_result=session)
        methods = [call["method"] for call in method_calls(proc)]
        self.assertEqual(methods.count("session/set_config_option"), 1)
        self.assertNotIn("session/set_model", methods)
        pin = next(call for call in method_calls(proc) if call["method"] == "session/set_config_option")
        self.assertEqual(
            pin["params"],
            {"sessionId": SESSION_ID, "configId": "model", "value": "grok-4.7"},
        )

    def test_whitespace_around_an_advertised_id_is_pinned(self) -> None:
        _completion, proc = self._drive(model=" \n grok-4.7 \t")
        pin = next(call for call in method_calls(proc) if call["method"] == "session/set_model")
        self.assertEqual(pin["params"]["modelId"], "grok-4.7")

    def test_empty_model_is_a_visible_error_without_prompt(self) -> None:
        for model in ("", "   "):
            with self.subTest(model=model):
                seen: list[str] = []
                with self.assertRaises(RuntimeError) as caught:
                    self._drive(model=model, on_method=seen.append)
                self.assertIn("default", str(caught.exception))
                self.assertIn("grok-4.7", str(caught.exception))
                self.assertNotIn("session/prompt", seen)
                self.assertNotIn("session/set_model", seen)

    def test_placeholder_model_is_a_visible_error_without_prompt(self) -> None:
        seen: list[str] = []

        def on_method(method: str) -> None:
            seen.append(method)

        with self.assertRaises(RuntimeError) as caught:
            self._drive(model="cursor-acp-unset", on_method=on_method)
        self.assertIn("default", str(caught.exception))
        self.assertIn("grok-4.7", str(caught.exception))
        self.assertNotIn("session/prompt", seen)
        self.assertNotIn("session/set_model", seen)
        self.assertNotIn("authenticate", seen)

    def test_unadvertised_auto_is_a_visible_error_without_sending_default(self) -> None:
        seen: list[str] = []
        with self.assertRaises(RuntimeError) as caught:
            self._drive(model="auto", on_method=seen.append)
        self.assertIn("default", str(caught.exception))
        self.assertIn("grok-4.7", str(caught.exception))
        self.assertNotIn("session/prompt", seen)
        self.assertNotIn("session/set_model", seen)
        self.assertNotIn("session/set_config_option", seen)

    def test_disabled_model_entry_is_a_visible_error_without_prompt(self) -> None:
        class Disabling(CursorACPClient):
            def model_entry_disabled(self, entry):
                return bool((entry.get("_meta") or {}).get("disabled"))

        client = Disabling(api_key=API_KEY)
        self.addCleanup(client.close)
        session = {
            "sessionId": SESSION_ID,
            "models": {
                "availableModels": [
                    {"modelId": "default", "name": "Auto"},
                    {"modelId": "grok-4.7", "name": "Grok", "_meta": {"disabled": True}},
                ]
            },
        }
        seen: list[str] = []
        with self.assertRaises(RuntimeError) as caught:
            self._drive(
                model="grok-4.7",
                client=client,
                session_result=session,
                on_method=seen.append,
            )
        self.assertIn("model entry is disabled", str(caught.exception))
        self.assertIn("default", str(caught.exception))
        self.assertNotIn("session/prompt", seen)

    def test_pin_request_error_includes_ids_and_does_not_prompt_again(self) -> None:
        seen: list[str] = []
        with self.assertRaises(RuntimeError) as caught:
            self._drive(model="grok-4.7", fail_method="session/set_model", on_method=seen.append)
        self.assertIn("default", str(caught.exception))
        self.assertIn("grok-4.7", str(caught.exception))
        self.assertEqual(seen.count("session/set_model"), 1)
        self.assertNotIn("session/prompt", seen)

    def test_usage_is_none_and_plain_text_finishes_stop(self) -> None:
        native = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "native-1",
                    "title": "Read",
                    "kind": "read",
                    "status": "completed",
                }
            },
        }

        def on_prompt():
            return [native, text_chunk("agent_message_chunk", "hello")]

        completion, _proc = self._drive(model="grok-4.7", on_prompt=on_prompt)
        self.assertTrue(hasattr(completion, "usage"))
        self.assertIsNone(completion.usage)
        self.assertEqual(completion.choices[0].finish_reason, "stop")
        self.assertEqual(completion.choices[0].message.content, "hello")
        self.assertEqual(completion.choices[0].message.tool_calls, [])

    def test_flat_tool_call_is_normalized_before_extract(self) -> None:
        completion, _proc = self._drive(
            model="grok-4.7",
            on_prompt=lambda: [text_chunk("agent_message_chunk", FLAT_CALL)],
        )
        message = completion.choices[0].message
        self.assertEqual(completion.choices[0].finish_reason, "tool_calls")
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].function.name, "canary_tool")
        self.assertEqual(json.loads(message.tool_calls[0].function.arguments), {"path": "canary.txt"})
        self.assertIsNone(completion.usage)

    def test_thought_chunks_stay_in_reasoning_and_are_not_tool_calls(self) -> None:
        def on_prompt():
            return [
                text_chunk("agent_thought_chunk", FLAT_CALL),
                text_chunk("agent_message_chunk", "visible answer"),
            ]

        completion, _proc = self._drive(model="grok-4.7", on_prompt=on_prompt)
        message = completion.choices[0].message
        self.assertEqual(completion.choices[0].finish_reason, "stop")
        self.assertEqual(message.tool_calls, [])
        self.assertEqual(message.content, "visible answer")
        self.assertEqual(message.reasoning, FLAT_CALL)
        self.assertEqual(message.reasoning_content, FLAT_CALL)

    def test_permission_request_is_reject_once(self) -> None:
        def on_prompt():
            return [
                server_request("session/request_permission", {"sessionId": SESSION_ID}, "perm-1"),
                text_chunk("agent_message_chunk", "hello"),
            ]

        _completion, proc = self._drive(model="grok-4.7", on_prompt=on_prompt)
        answers = [msg for msg in result_messages(proc) if msg.get("id") == "perm-1"]
        self.assertEqual(answers, [{"jsonrpc": "2.0", "id": "perm-1", "result": {"outcome": {"outcome": "reject-once"}}}])
        wire = "".join(proc.written)
        self.assertNotIn("allow-always", wire)
        self.assertNotIn("allow-once", wire)
        self.assertNotIn("cancelled", wire)
        self.assertNotIn('"selected"', wire)

    def test_fs_methods_return_method_not_found_without_touching_the_path(self) -> None:
        canary = Path(tempfile.mkdtemp(prefix="cursor-acp-fs-")) / "canary.txt"
        self.addCleanup(lambda: __import__("shutil").rmtree(canary.parent))
        canary.write_text("fs-canary-contents")
        target = canary.parent / "written.txt"

        def on_prompt():
            return [
                server_request("fs/read_text_file", {"path": str(canary)}, "fs-read"),
                server_request(
                    "fs/write_text_file",
                    {"path": str(target), "content": "pwned"},
                    "fs-write",
                ),
                text_chunk("agent_message_chunk", "hello"),
            ]

        _completion, proc = self._drive(model="grok-4.7", on_prompt=on_prompt)
        by_id = {msg.get("id"): msg for msg in result_messages(proc)}
        self.assertEqual(by_id["fs-read"]["error"]["code"], -32601)
        self.assertEqual(by_id["fs-write"]["error"]["code"], -32601)
        self.assertNotIn("fs-canary-contents", "".join(proc.written))
        self.assertEqual(canary.read_text(), "fs-canary-contents")
        self.assertFalse(target.exists())

    def test_terminal_methods_return_method_not_found(self) -> None:
        def on_prompt():
            return [
                server_request("terminal/create", {"command": "touch /tmp/hca-should-not-run"}, "term-1"),
                server_request("terminal/output", {"terminalId": "t"}, "term-2"),
                server_request("terminal/kill", {"terminalId": "t"}, "term-3"),
                text_chunk("agent_message_chunk", "hello"),
            ]

        _completion, proc = self._drive(model="grok-4.7", on_prompt=on_prompt)
        by_id = {msg.get("id"): msg for msg in result_messages(proc)}
        self.assertEqual(by_id["term-1"]["error"]["code"], -32601)
        self.assertEqual(by_id["term-2"]["error"]["code"], -32601)
        self.assertEqual(by_id["term-3"]["error"]["code"], -32601)
        self.assertFalse(Path("/tmp/hca-should-not-run").exists())

    def test_unknown_method_keeps_the_stock_error(self) -> None:
        def on_prompt():
            return [
                server_request("cursor/ask_question", {"question": "pick"}, "ask-1"),
                text_chunk("agent_message_chunk", "hello"),
            ]

        _completion, proc = self._drive(model="grok-4.7", on_prompt=on_prompt)
        answer = next(msg for msg in result_messages(proc) if msg.get("id") == "ask-1")
        self.assertEqual(
            answer["error"],
            {
                "code": -32601,
                "message": "ACP client method 'cursor/ask_question' is not supported by Hermes yet.",
            },
        )

    def test_successful_prompt_deletes_session_residue_and_tmpdir(self) -> None:
        session_canary, tmp_canary, outside = self._plant_residue()

        def on_method(method: str) -> None:
            if method == "session/new":
                session_canary.parent.mkdir(parents=True, exist_ok=True)
                session_canary.write_text("session-residue")
                tmp_canary.write_text("tmp-residue")

        self._drive(model="grok-4.7", on_method=on_method)
        self.assertFalse(session_canary.exists())
        self.assertFalse(tmp_canary.exists())
        self.assertFalse(self.client._fence.tmpdir.exists())
        self.assertEqual(outside.read_text(), "outside-residue")

    def test_pin_error_still_deletes_session_residue(self) -> None:
        session_canary, tmp_canary, outside = self._plant_residue()

        def on_method(method: str) -> None:
            if method == "session/new":
                session_canary.parent.mkdir(parents=True, exist_ok=True)
                session_canary.write_text("session-residue")
                tmp_canary.write_text("tmp-residue")

        with self.assertRaises(RuntimeError):
            self._drive(model="auto", on_method=on_method)
        self.assertFalse(session_canary.exists())
        self.assertFalse(tmp_canary.exists())
        self.assertEqual(outside.read_text(), "outside-residue")

    def test_set_mode_error_still_deletes_session_residue(self) -> None:
        session_canary, tmp_canary, _outside = self._plant_residue()

        def on_method(method: str) -> None:
            if method == "session/new":
                session_canary.parent.mkdir(parents=True, exist_ok=True)
                session_canary.write_text("session-residue")
                tmp_canary.write_text("tmp-residue")

        seen: list[str] = []

        def watch(method: str) -> None:
            seen.append(method)
            on_method(method)

        with self.assertRaises(RuntimeError):
            self._drive(model="grok-4.7", fail_method="session/set_mode", on_method=watch)
        self.assertNotIn("session/prompt", seen)
        self.assertFalse(session_canary.exists())
        self.assertFalse(tmp_canary.exists())

    def test_initialize_error_keeps_tmpdir_when_no_session_exists(self) -> None:
        tmp_canary = self.client._fence.tmpdir / "residue.txt"
        tmp_canary.write_text("tmp-residue")
        with self.assertRaises(RuntimeError):
            self._drive(model="grok-4.7", fail_method="initialize")
        self.assertTrue(tmp_canary.is_file())
        self.assertEqual(tmp_canary.read_text(), "tmp-residue")

    def test_second_turn_prepares_the_fence_again(self) -> None:
        _completion, first = self._drive(model="grok-4.7")
        self.assertTrue(first.saw_cli)
        cli = Path(self.client._acp_cwd) / ".cursor" / "cli.json"
        cli.unlink()
        _completion, second = self._drive(model="grok-4.7")
        self.assertTrue(second.saw_cli)

    def test_close_deletes_the_fence_and_does_not_prepare_again(self) -> None:
        cwd = Path(self.client._acp_cwd)
        self.client.close()
        self.assertFalse(cwd.exists())
        calls: list[str] = []
        self.client._fence.prepare_spawn = lambda: calls.append("prepare")
        self.client._fence.cleanup_turn = lambda session_id: calls.append("cleanup")
        with mock.patch(
            "cursor_acp.client.subprocess.Popen",
            side_effect=AssertionError("closed client spawned"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                self.client.chat.completions.create(
                    model="grok-4.7",
                    messages=[{"role": "user", "content": "ping-sentinel"}],
                    timeout=3,
                )
        self.assertEqual(calls, [])
        self.assertNotIn("fence is closed", str(caught.exception))

    def test_cancel_terminates_only_the_request_child(self) -> None:
        procs: list[FakeProcess] = []

        def popen(argv, **kwargs):
            proc = FakeProcess(lambda msg: [])
            procs.append(proc)
            return proc

        with mock.patch("cursor_acp.client.subprocess.Popen", side_effect=popen):
            first = self.client._spawn()
            second = self.client._spawn()
        self.client.cancel()
        self.assertTrue(second.terminated)
        self.assertFalse(second.killed)
        self.assertFalse(first.terminated)
        self.assertFalse(first.killed)
        self.assertNotIn("session/cancel", "".join(first.written + second.written))

    def test_cancel_kills_after_terminate_times_out_without_walking_siblings(self) -> None:
        class SlowDeath:
            def __init__(self) -> None:
                self.terminated = False
                self.killed = False
                self.stdin = io.StringIO()

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("agent", timeout if timeout is not None else 2)

            def kill(self) -> None:
                self.killed = True

        request = SlowDeath()
        sibling = SlowDeath()
        self.client._request_process = request
        self.client._active_processes.add(sibling)
        self.client.cancel()
        self.assertTrue(request.terminated)
        self.assertTrue(request.killed)
        self.assertFalse(sibling.terminated)
        self.assertFalse(sibling.killed)
        self.assertNotIn("session/cancel", request.stdin.getvalue())

    def test_stream_response_keeps_usage_none(self) -> None:
        chunks, _proc = self._drive(model="grok-4.7", stream=True)
        self.assertIsNone(chunks[-1].usage)
        self.assertEqual(chunks[0].choices[0].finish_reason, "stop")

    def test_overlapping_prompts_clean_only_their_own_session(self) -> None:
        """Two live sessions: each prompt deletes its own id and leaves the other."""
        sessions = self.client._fence.config_dir / "acp-sessions"
        ids = ["sess-live-a", "sess-live-b"]
        id_lock = threading.Lock()
        id_box = {"n": 0}
        release = {sid: threading.Event() for sid in ids}
        arrived = {sid: threading.Event() for sid in ids}
        for event in release.values():
            self.addCleanup(event.set)

        def handle(proc: FakeProcess, msg: dict) -> list[dict]:
            method = msg["method"]
            if method == "initialize":
                return [rpc_result(msg, {"protocolVersion": 1})]
            if method == "session/new":
                with id_lock:
                    sid = ids[id_box["n"]]
                    id_box["n"] += 1
                proc.session_id = sid
                residue = sessions / sid / "residue.txt"
                residue.parent.mkdir(parents=True, exist_ok=True)
                residue.write_text(sid)
                return [
                    rpc_result(
                        msg,
                        {
                            "sessionId": sid,
                            "models": {
                                "currentModelId": "grok-4.7",
                                "availableModels": [{"modelId": "grok-4.7", "name": "Grok"}],
                            },
                        },
                    )
                ]
            if method in {"session/set_mode", "session/set_model"}:
                return [rpc_result(msg, {})]
            if method == "session/prompt":
                sid = proc.session_id
                arrived[sid].set()
                self.assertTrue(release[sid].wait(timeout=5))
                return [
                    text_chunk("agent_message_chunk", sid),
                    rpc_result(msg, {"stopReason": "end_turn"}),
                ]
            return [rpc_error(msg, -32601, f"unexpected {method}")]

        def popen(argv, **kwargs):
            proc = FakeProcess(lambda msg: handle(proc, msg))
            proc.session_id = ""
            return proc

        errors: list[BaseException] = []

        def complete(text: str) -> None:
            try:
                self.client.chat.completions.create(
                    model="grok-4.7",
                    messages=[{"role": "user", "content": text}],
                    timeout=5,
                )
            except BaseException as exc:
                errors.append(exc)

        with mock.patch("cursor_acp.client.subprocess.Popen", side_effect=popen):
            first = threading.Thread(target=complete, args=("prompt-a",))
            second = threading.Thread(target=complete, args=("prompt-b",))
            first.start()
            self.addCleanup(first.join, 5)
            self.assertTrue(arrived["sess-live-a"].wait(timeout=5))
            second.start()
            self.addCleanup(second.join, 5)
            self.assertTrue(arrived["sess-live-b"].wait(timeout=5))
            self.assertTrue((sessions / "sess-live-a" / "residue.txt").is_file())
            self.assertTrue((sessions / "sess-live-b" / "residue.txt").is_file())

            release["sess-live-b"].set()
            second.join(5)
            self.assertFalse(second.is_alive())
            self.assertFalse((sessions / "sess-live-b").exists())
            self.assertTrue((sessions / "sess-live-a" / "residue.txt").is_file())

            release["sess-live-a"].set()
            first.join(5)
            self.assertFalse(first.is_alive())

        self.assertEqual(errors, [])
        self.assertFalse((sessions / "sess-live-a").exists())
        self.assertFalse((sessions / "sess-live-b").exists())
