"""Proof oracles for chat, Hermes read_file, native Read, and the model pin.

The fake stdio process never execs ``agent``. These checks describe the client
already on this branch. They do not read ``~/.hermes`` or ``~/.cursor``.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cursor_acp.client import CursorACPClient
from cursor_acp.model_pin import ModelPinError

try:
    from tests.test_client import (
        API_KEY,
        SESSION_ID,
        FakeProcess,
        method_calls,
        result_messages,
        rpc_error,
        rpc_result,
        server_request,
        text_chunk,
    )
except ModuleNotFoundError:  # unittest discover -s tests loads modules from that directory
    from test_client import (
        API_KEY,
        SESSION_ID,
        FakeProcess,
        method_calls,
        result_messages,
        rpc_error,
        rpc_result,
        server_request,
        text_chunk,
    )

REQUESTED_MODEL = "grok-4.7"
ADVERTISED = ("default", "grok-4.7")
ABSENT_MODEL = "not-in-session-list"
CANARY_PATH = "/var/hca-proof-canary/token.txt"
SENTINEL = "HCA-SENTINEL-PROOF-6"
EMPTY_PROMPT_TEXT = "No prompt content provided."
STORED_TOOL_BODY = "stored-by-executor"
USER_TEXT = "Read the file named by the tool."


def chat_turn_failures(completion, proc: FakeProcess, *, model: str, reasoning: str) -> list[str]:
    """Reasons a no-tool turn fails the chat oracle. Empty means it passes."""
    failures: list[str] = []
    choice = completion.choices[0]
    message = choice.message
    content = message.content or ""
    if content == EMPTY_PROMPT_TEXT:
        failures.append("only content is No prompt content provided.")
    if choice.finish_reason != "stop":
        failures.append(f"finish_reason is {choice.finish_reason!r}")
    if message.tool_calls != []:
        failures.append(f"tool_calls is {message.tool_calls!r}")
    if "CURSOR-ACP-OK" not in content:
        failures.append("content does not contain CURSOR-ACP-OK")
    if (message.reasoning or "") != reasoning or (message.reasoning_content or "") != reasoning:
        failures.append(
            f"reasoning is {message.reasoning!r} / {message.reasoning_content!r}, want {reasoning!r}"
        )
    if completion.model != model:
        failures.append(f"model is {completion.model!r}")
    if completion.usage is not None:
        failures.append(f"usage is {completion.usage!r}")
    pins = [
        call
        for call in method_calls(proc)
        if call["method"] in {"session/set_model", "session/set_config_option"}
        and model in json.dumps(call.get("params") or {})
    ]
    if not pins:
        failures.append("fake did not receive session/set_model or session/set_config_option for the id")
    return failures


def native_read_verdict(updates: list[dict], assistant_text: str, sentinel: str) -> str:
    """``pass``, ``fail``, or ``invalidated`` for one native-read content check."""
    reads = [update for update in updates if update.get("kind") == "read"]
    if "<tool_call>" in assistant_text and not reads:
        return "invalidated"
    raw_text = "\n".join(_raw_output_text(update.get("rawOutput")) for update in reads)
    if sentinel in raw_text or sentinel in assistant_text:
        return "fail"
    if reads:
        return "pass"
    return "invalidated"


def _raw_output_text(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        content = raw.get("content")
        dumped = json.dumps(raw, ensure_ascii=False)
        if isinstance(content, str):
            return content + "\n" + dumped
        return dumped
    if raw is None:
        return ""
    return str(raw)


def prompt_text(proc: FakeProcess) -> str:
    prompts = [call for call in method_calls(proc) if call["method"] == "session/prompt"]
    blocks = prompts[-1]["params"]["prompt"]
    return "\n".join(str(block.get("text") or "") for block in blocks)


def permission_outcomes(proc: FakeProcess) -> list[str]:
    outcomes: list[str] = []
    for msg in result_messages(proc):
        outcome = (msg.get("result") or {}).get("outcome") if isinstance(msg.get("result"), dict) else None
        if isinstance(outcome, dict) and "outcome" in outcome:
            outcomes.append(outcome["outcome"])
    return outcomes


class ProofOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        with mock.patch(
            "cursor_acp.client.subprocess.Popen",
            side_effect=AssertionError("constructor started a process"),
        ):
            self.client = CursorACPClient(
                api_key=API_KEY,
                command="/no/such/cursor-agent-hca",
                args=["acp"],
            )
        self.addCleanup(self._retire, self.client)

    def _retire(self, client: CursorACPClient) -> None:
        fence = getattr(client, "_fence", None)
        try:
            client.close()
        finally:
            if fence is not None and not fence._closed:
                fence.close()

    def _complete(
        self,
        procs: list[FakeProcess],
        *,
        model: str,
        messages: list[dict],
        on_prompt,
        available: tuple[str, ...] = ADVERTISED,
        session_result: dict | None = None,
        tools=None,
    ):
        def handle(msg: dict) -> list[dict]:
            method = msg["method"]
            if method == "initialize":
                return [rpc_result(msg, {"protocolVersion": 1})]
            if method == "session/new":
                payload = session_result if session_result is not None else {
                    "sessionId": SESSION_ID,
                    "models": {
                        "currentModelId": available[0] if available else "",
                        "availableModels": [{"modelId": model_id, "name": model_id} for model_id in available],
                    },
                }
                return [rpc_result(msg, payload)]
            if method in {"session/set_mode", "session/set_model", "session/set_config_option"}:
                return [rpc_result(msg, {})]
            if method == "session/prompt":
                outbound = list(on_prompt())
                outbound.append(rpc_result(msg, {"stopReason": "end_turn"}))
                return outbound
            return [rpc_error(msg, -32601, f"unexpected {method}")]

        def popen(argv, **kwargs):
            proc = FakeProcess(handle)
            procs.append(proc)
            return proc

        with mock.patch("cursor_acp.client.subprocess.Popen", side_effect=popen):
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                timeout=3,
            )

    def test_chat_turn_without_tools_passes_when_the_reply_contains_the_marker(self) -> None:
        thoughts = ("plan the reply ", "then stop")
        messages = ("Preparing ", "CURSOR-ACP-OK")
        procs: list[FakeProcess] = []

        def on_prompt():
            return [
                text_chunk("agent_thought_chunk", thoughts[0]),
                text_chunk("agent_message_chunk", messages[0]),
                text_chunk("agent_thought_chunk", thoughts[1]),
                text_chunk("agent_message_chunk", messages[1]),
            ]

        completion = self._complete(
            procs,
            model=REQUESTED_MODEL,
            messages=[{"role": "user", "content": "Reply with the marker."}],
            on_prompt=on_prompt,
        )

        self.assertEqual(completion.choices[0].message.content, "Preparing CURSOR-ACP-OK")
        self.assertEqual(chat_turn_failures(completion, procs[0], model=REQUESTED_MODEL, reasoning="".join(thoughts)), [])
        self.assertEqual(
            [call["method"] for call in method_calls(procs[0]) if call["method"] == "session/set_model"],
            ["session/set_model"],
        )
        pin = next(call for call in method_calls(procs[0]) if call["method"] == "session/set_model")
        self.assertEqual(pin["params"]["modelId"], REQUESTED_MODEL)

    def test_chat_turn_fails_when_the_only_content_is_the_cli_empty_prompt(self) -> None:
        procs: list[FakeProcess] = []
        completion = self._complete(
            procs,
            model=REQUESTED_MODEL,
            messages=[{"role": "user", "content": "Reply with the marker."}],
            on_prompt=lambda: [text_chunk("agent_message_chunk", EMPTY_PROMPT_TEXT)],
        )

        failures = chat_turn_failures(completion, procs[0], model=REQUESTED_MODEL, reasoning="")
        self.assertIn("only content is No prompt content provided.", failures)

    def test_hermes_read_file_keeps_the_sentinel_off_the_fs_bridge(self) -> None:
        self.assertNotIn(CANARY_PATH, USER_TEXT)
        self.assertNotIn(SENTINEL, USER_TEXT)
        canary = Path(tempfile.mkdtemp(prefix="cursor-acp-proof-")) / "token.txt"
        self.addCleanup(lambda: __import__("shutil").rmtree(canary.parent))
        canary.write_text(SENTINEL, encoding="utf-8")
        flat = (
            '<tool_call>{"name":"read_file","arguments":{"path":"'
            + CANARY_PATH
            + '"}}</tool_call>'
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        procs: list[FakeProcess] = []

        def on_prompt():
            return [
                server_request("session/request_permission", {"sessionId": SESSION_ID}, "perm-1"),
                server_request("fs/read_text_file", {"path": str(canary)}, "fs-read"),
                text_chunk("agent_message_chunk", flat),
            ]

        completion = self._complete(
            procs,
            model=REQUESTED_MODEL,
            messages=[{"role": "user", "content": USER_TEXT}],
            on_prompt=on_prompt,
            tools=tools,
        )
        first = procs[0]
        first_text = prompt_text(first)
        call = completion.choices[0].message.tool_calls[0]

        self.assertIn("read_file", first_text)
        self.assertNotIn(SENTINEL, first_text)
        self.assertNotIn(CANARY_PATH, first_text)
        self.assertEqual(call.function.name, "read_file")
        self.assertEqual(json.loads(call.function.arguments), {"path": CANARY_PATH})
        self.assertNotIn(SENTINEL, completion.choices[0].message.content or "")
        self.assertEqual(permission_outcomes(first), ["reject-once"])
        fs_reply = next(msg for msg in result_messages(first) if msg.get("id") == "fs-read")
        self.assertIn("error", fs_reply)
        self.assertNotIn(SENTINEL, json.dumps(fs_reply))
        self.assertNotIn(SENTINEL, "".join(first.written))
        self.assertEqual(canary.read_text(encoding="utf-8"), SENTINEL)

        stored = {
            "id": call.id,
            "type": call.type,
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
        follow = self._complete(
            procs,
            model=REQUESTED_MODEL,
            messages=[
                {"role": "user", "content": USER_TEXT},
                {"role": "assistant", "content": completion.choices[0].message.content or "", "tool_calls": [stored]},
                {
                    "role": "tool",
                    "name": "read_file",
                    "tool_call_id": call.id,
                    "content": STORED_TOOL_BODY,
                },
            ],
            on_prompt=on_prompt,
            tools=tools,
        )
        second_text = prompt_text(procs[1])

        self.assertEqual(stored["function"]["name"], "read_file")
        self.assertIn("Tool:\n" + STORED_TOOL_BODY, second_text)
        self.assertNotIn(SENTINEL, "".join(procs[1].written))
        self.assertEqual(permission_outcomes(procs[1]), ["reject-once"])
        self.assertIsNone(follow.usage)

    def test_native_read_without_the_sentinel_passes_the_content_check(self) -> None:
        assistant = f"The path {CANARY_PATH} was refused."
        update = {
            "sessionUpdate": "tool_call",
            "toolCallId": "native-read-1",
            "kind": "read",
            "status": "completed",
            "rawOutput": {"content": "Error: Permission denied"},
        }
        procs: list[FakeProcess] = []
        completion = self._complete(
            procs,
            model=REQUESTED_MODEL,
            messages=[{"role": "user", "content": f"Look at {CANARY_PATH}"}],
            on_prompt=lambda: [_session_update(update), text_chunk("agent_message_chunk", assistant)],
        )

        verdict = native_read_verdict([update], assistant, SENTINEL)
        self.assertEqual(verdict, "pass")
        self.assertEqual(completion.choices[0].message.content, assistant)
        self.assertNotIn(SENTINEL, completion.choices[0].message.content)
        self.assertEqual(completion.choices[0].message.tool_calls, [])
        self.assertEqual(completion.choices[0].finish_reason, "stop")

    def test_native_read_raw_output_with_the_sentinel_fails_the_content_check(self) -> None:
        assistant = f"The path {CANARY_PATH} was refused."
        update = {
            "sessionUpdate": "tool_call",
            "toolCallId": "native-read-1",
            "kind": "read",
            "status": "completed",
            "rawOutput": {"content": SENTINEL},
        }
        procs: list[FakeProcess] = []
        completion = self._complete(
            procs,
            model=REQUESTED_MODEL,
            messages=[{"role": "user", "content": f"Look at {CANARY_PATH}"}],
            on_prompt=lambda: [_session_update(update), text_chunk("agent_message_chunk", assistant)],
        )

        verdict = native_read_verdict([update], assistant, SENTINEL)
        self.assertEqual(verdict, "fail")
        self.assertNotEqual(verdict, "pass")
        self.assertNotIn(SENTINEL, completion.choices[0].message.content or "")

    def test_hermes_tool_call_without_a_native_read_invalidates_the_proof(self) -> None:
        model_text = (
            '<tool_call>{"name":"read_file","arguments":{"path":"'
            + CANARY_PATH
            + '"}}</tool_call>'
        )
        procs: list[FakeProcess] = []
        completion = self._complete(
            procs,
            model=REQUESTED_MODEL,
            messages=[{"role": "user", "content": USER_TEXT}],
            on_prompt=lambda: [text_chunk("agent_message_chunk", model_text)],
        )

        verdict = native_read_verdict([], model_text, SENTINEL)
        self.assertEqual(verdict, "invalidated")
        self.assertNotEqual(verdict, "pass")
        self.assertEqual(completion.choices[0].message.tool_calls[0].function.name, "read_file")
        self.assertNotIn("<tool_call>", completion.choices[0].message.content or "")

    def test_absent_model_raises_model_pin_error_before_session_prompt(self) -> None:
        procs: list[FakeProcess] = []
        with self.assertRaises(ModelPinError) as caught:
            self._complete(
                procs,
                model=ABSENT_MODEL,
                messages=[{"role": "user", "content": "Reply with the marker."}],
                on_prompt=lambda: [text_chunk("agent_message_chunk", "CURSOR-ACP-OK")],
            )

        message = str(caught.exception)
        for model_id in ADVERTISED:
            self.assertIn(model_id, message)
        self.assertEqual(len(procs), 1)
        methods = [call["method"] for call in method_calls(procs[0])]
        self.assertNotIn("session/prompt", methods)
        for blob in procs[0].written:
            for line in blob.splitlines():
                if line.strip():
                    self.assertNotIn("session/prompt", line)


def _session_update(update: dict) -> dict:
    return {"jsonrpc": "2.0", "method": "session/update", "params": {"update": update}}


if __name__ == "__main__":
    unittest.main()
