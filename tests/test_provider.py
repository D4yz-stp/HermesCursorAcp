"""Provider registration: Hermes discovers cursor-acp and lists models from one short session."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = "/home/d4yz_/.hermes/hermes-agent"
for _path in (HERMES_ROOT, str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cursor_acp.client import CursorACPClient

SESSION_ID = "sess-fetch-hca"
LIVE_IDS = ["default", "grok-4.7"]
PLUGIN_LINES = [
    "name: cursor-acp",
    "kind: model-provider",
    "description: Hermes client for the Cursor agent ACP process.",
]
SUBCLASS_INITIALIZE = {
    "protocolVersion": 1,
    "clientCapabilities": {
        "fs": {"readTextFile": False, "writeTextFile": False},
        "terminal": False,
        "_meta": {"parameterizedModelPicker": True},
    },
    "clientInfo": {"name": "hermes-agent", "title": "Hermes Agent", "version": "0.0.0"},
}


def rpc_result(msg, result):
    return {"jsonrpc": "2.0", "id": msg["id"], "result": result}


def rpc_error(msg, code, message):
    return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": code, "message": message}}


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
        self.returncode = None
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
        self.returncode = -15
        self.stdout.close()
        self.stderr.close()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("agent", timeout if timeout is not None else 2)
        return self.returncode


def method_calls(proc: FakeProcess) -> list[dict]:
    messages = []
    for blob in proc.written:
        for line in blob.splitlines():
            if line.strip():
                messages.append(json.loads(line))
    return [msg for msg in messages if "method" in msg]


class ProviderRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="cursor-acp-home-")
        self.home = Path(self._tmpdir.name)
        self._old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.home)
        self.plugin_dir = self.home / "plugins" / "model-providers" / "cursor-acp"
        self.plugin_dir.mkdir(parents=True)
        for name in ("__init__.py", "plugin.yaml"):
            source = ROOT / "cursor_acp" / name
            if source.is_file():
                shutil.copyfile(source, self.plugin_dir / name)
        self.addCleanup(self._restore_home)
        self.addCleanup(self._tmpdir.cleanup)
        from providers import get_provider_profile, plugin_load_error, provider_source

        self.profile = get_provider_profile("cursor-acp")
        self.load_error = plugin_load_error("cursor-acp")
        self.source = provider_source("cursor-acp")

    def _restore_home(self) -> None:
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home

    def test_discovery_registers_the_external_process_profile(self) -> None:
        self.assertIsNotNone(self.profile, self.load_error)
        self.assertEqual(self.source, "user")
        self.assertEqual(self.profile.name, "cursor-acp")
        self.assertEqual(self.profile.auth_type, "external_process")
        self.assertEqual(self.profile.api_mode, "chat_completions")
        self.assertEqual(self.profile.process_command, "agent")
        self.assertEqual(self.profile.process_args, ("acp",))
        self.assertEqual(self.profile.static_models, ())
        self.assertEqual(self.profile.fallback_models, ())
        self.assertEqual(self.profile.base_url, "acp://cursor")
        self.assertFalse(self.profile.supports_vision)
        self.assertIsNone(self.profile.get_model_context_length("grok-4.7"))
        self.assertFalse((self.home / "plugins" / "cursor-acp").exists())
        self.assertFalse((Path.home() / ".hermes" / "plugins" / "model-providers" / "cursor-acp").exists())

    def test_plugin_manifest_is_the_three_declared_lines(self) -> None:
        manifest = self.plugin_dir / "plugin.yaml"
        text = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
        self.assertEqual(text.splitlines(), PLUGIN_LINES)

    def test_create_client_returns_cursor_client_and_accepts_unknown_keys(self) -> None:
        self.assertIsNotNone(self.profile, self.load_error)
        client = self.profile.create_client(
            api_key="provider-key-hca",
            base_url="acp://cursor",
            command="agent",
            args=["acp"],
            timeout=3,
            not_a_cursor_kw="kept",
        )
        try:
            self.assertIsInstance(client, CursorACPClient)
            temp_root = Path(tempfile.gettempdir()).resolve()
            self.assertTrue(client._fence.config_dir.resolve().is_relative_to(temp_root))
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

    def test_fetch_models_returns_live_ids_from_one_subclass_session(self) -> None:
        self.assertIsNotNone(self.profile, self.load_error)
        planted: dict[str, Path] = {}
        procs: list[FakeProcess] = []

        def handle(msg):
            method = msg["method"]
            if method == "initialize":
                return [rpc_result(msg, {"protocolVersion": 1})]
            if method == "session/new":
                return [rpc_result(msg, _session_payload())]
            if method == "session/set_mode":
                return [rpc_result(msg, {})]
            return [rpc_error(msg, -32601, f"unexpected {method}")]

        def popen(argv, **kwargs):
            env = kwargs.get("env") or {}
            config = Path(env["CURSOR_CONFIG_DIR"])
            temp_root = Path(tempfile.gettempdir()).resolve()
            if not config.resolve().is_relative_to(temp_root):
                raise AssertionError(f"refusing to plant a session under {config}")
            session_dir = config / "acp-sessions" / SESSION_ID
            session_dir.mkdir(parents=True)
            (session_dir / "residue.txt").write_text("session-residue")
            planted["dir"] = session_dir
            proc = FakeProcess(handle)
            procs.append(proc)
            return proc

        with mock.patch("cursor_acp.client.subprocess.Popen", side_effect=popen), mock.patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            side_effect=AssertionError("fetch_models opened a network catalog"),
        ):
            found = self.profile.fetch_models(api_key=None, base_url="acp://cursor", timeout=3)

        self.assertEqual(found, LIVE_IDS)
        self.assertEqual(len(procs), 1)
        calls = method_calls(procs[0])
        self.assertEqual([call["method"] for call in calls], ["initialize", "session/new", "session/set_mode"])
        self.assertEqual(calls[0]["params"], SUBCLASS_INITIALIZE)
        self.assertIn("dir", planted)
        self.assertFalse(planted["dir"].exists())

    def test_fetch_models_raises_and_removes_the_session_directory(self) -> None:
        self.assertIsNotNone(self.profile, self.load_error)
        planted: dict[str, Path] = {}

        def handle(msg):
            method = msg["method"]
            if method == "initialize":
                return [rpc_result(msg, {"protocolVersion": 1})]
            if method == "session/new":
                return [rpc_result(msg, _session_payload())]
            if method == "session/set_mode":
                return [rpc_error(msg, -32603, "mode failed")]
            return [rpc_error(msg, -32601, f"unexpected {method}")]

        def popen(argv, **kwargs):
            env = kwargs.get("env") or {}
            config = Path(env["CURSOR_CONFIG_DIR"])
            temp_root = Path(tempfile.gettempdir()).resolve()
            if not config.resolve().is_relative_to(temp_root):
                raise AssertionError(f"refusing to plant a session under {config}")
            session_dir = config / "acp-sessions" / SESSION_ID
            session_dir.mkdir(parents=True)
            (session_dir / "residue.txt").write_text("session-residue")
            planted["dir"] = session_dir
            return FakeProcess(handle)

        with mock.patch("cursor_acp.client.subprocess.Popen", side_effect=popen), mock.patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            side_effect=AssertionError("fetch_models opened a network catalog"),
        ):
            with self.assertRaises(RuntimeError):
                self.profile.fetch_models()

        self.assertIn("dir", planted)
        self.assertFalse(planted["dir"].exists())
        self.assertNotEqual(getattr(self.profile, "fallback_models", None), LIVE_IDS)


def _session_payload() -> dict:
    return {
        "sessionId": SESSION_ID,
        "models": {
            "currentModelId": "default",
            "availableModels": [
                {"modelId": "default", "name": "Auto"},
                {"modelId": "grok-4.7", "name": "Grok 4.7"},
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
