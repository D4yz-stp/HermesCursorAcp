"""Tests for rewriting flat <tool_call> objects into Hermes function calls."""

import json
import sys
import unittest

HERMES_ROOT = "/home/d4yz_/.hermes/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

from agent.acp_openai_bridge import extract_tool_calls_from_text
from cursor_acp.normalize import normalize_tool_call_text

FLAT_CANARY = (
    '<tool_call>{"name":"canary_tool","arguments":{"path":"canary.txt"}}</tool_call>'
)
SHAPED_CANARY = (
    '<tool_call>{"type":"function","function":{"name":"canary_tool",'
    '"arguments":"{\\"path\\":\\"canary.txt\\"}"}}</tool_call>'
)


class NormalizeToolCallTextTests(unittest.TestCase):
    def test_flat_object_becomes_function_call_with_compact_argument_string(self):
        self.assertEqual(normalize_tool_call_text(FLAT_CANARY), SHAPED_CANARY)

    def test_spaced_object_arguments_are_dumped_without_spaces(self):
        text = '<tool_call>{"name": "canary_tool", "arguments": {"path": "canary.txt"}}</tool_call>'
        self.assertEqual(normalize_tool_call_text(text), SHAPED_CANARY)

    def test_string_arguments_stay_that_string(self):
        text = (
            '<tool_call>{"name":"canary_tool","arguments":"{\\"path\\":\\"canary.txt\\"}"}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), SHAPED_CANARY)

    def test_plain_string_arguments_are_not_reencoded(self):
        text = '<tool_call>{"name":"canary_tool","arguments":"hello"}</tool_call>'
        expected = (
            '<tool_call>{"type":"function","function":{"name":"canary_tool",'
            '"arguments":"hello"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_missing_arguments_become_empty_object_string(self):
        text = '<tool_call>{"name":"canary_tool"}</tool_call>'
        expected = (
            '<tool_call>{"type":"function","function":{"name":"canary_tool",'
            '"arguments":"{}"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_array_arguments_are_dumped_without_spaces(self):
        text = '<tool_call>{"name":"canary_tool","arguments":[1,{"k":"v"}]}</tool_call>'
        expected = (
            '<tool_call>{"type":"function","function":{"name":"canary_tool",'
            '"arguments":"[1,{\\"k\\":\\"v\\"}]"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_inner_whitespace_still_rewrites_the_object(self):
        text = '<tool_call>\n  {"name": "canary_tool"} \n</tool_call>'
        expected = (
            '<tool_call>{"type":"function","function":{"name":"canary_tool",'
            '"arguments":"{}"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_name_with_quote_is_json_escaped(self):
        text = '<tool_call>{"name":"canary\\"tool","arguments":{}}</tool_call>'
        expected = (
            '<tool_call>{"type":"function","function":{"name":"canary\\"tool",'
            '"arguments":"{}"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_blank_name_is_still_rewritten(self):
        text = '<tool_call>{"name":"  ","arguments":{}}</tool_call>'
        expected = (
            '<tool_call>{"type":"function","function":{"name":"  ",'
            '"arguments":"{}"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_object_that_already_has_function_is_unchanged(self):
        text = (
            '<tool_call>{"name":"canary_tool","function":{"name":"canary_tool",'
            '"arguments":"{}"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_function_key_with_null_value_is_unchanged(self):
        text = '<tool_call>{"name":"canary_tool","function":null}</tool_call>'
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_non_string_name_is_unchanged(self):
        text = '<tool_call>{"name":1,"arguments":{}}</tool_call>'
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_extra_keys_are_dropped(self):
        text = '<tool_call>{"id":"keep-me","name":"canary_tool","arguments":{}}</tool_call>'
        expected = (
            '<tool_call>{"type":"function","function":{"name":"canary_tool",'
            '"arguments":"{}"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_object_without_name_is_unchanged(self):
        text = '<tool_call>{"arguments":{"path":"canary.txt"}}</tool_call>'
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_invalid_json_is_unchanged(self):
        text = "before <tool_call>{not json}</tool_call> after"
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_single_quoted_pseudo_json_is_unchanged(self):
        text = (
            "<tool_call>{'name': 'canary_tool', 'arguments': {'path': 'canary.txt'}}</tool_call>"
        )
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_arg_key_dialect_is_unchanged(self):
        text = "<tool_call>canary_tool<arg_key>path</arg_key>canary.txt</tool_call>"
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_array_body_is_unchanged(self):
        text = '<tool_call>[{"name":"canary_tool","arguments":{}}]</tool_call>'
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_empty_block_is_unchanged(self):
        text = "<tool_call></tool_call>"
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_uppercase_tag_is_unchanged(self):
        text = '<TOOL_CALL>{"name":"canary_tool","arguments":{}}</TOOL_CALL>'
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_text_outside_tags_is_unchanged(self):
        text = '  keep\n<tool_call>{"name":"canary_tool"}</tool_call>\nkeep  '
        expected = (
            '  keep\n<tool_call>{"type":"function","function":{"name":"canary_tool",'
            '"arguments":"{}"}}</tool_call>\nkeep  '
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_think_tags_are_not_stripped(self):
        text = (
            '<think>leave this</think>\n'
            '<tool_call>{"name":"canary_tool","arguments":{"path":"canary.txt"}}</tool_call>'
        )
        expected = "<think>leave this</think>\n" + SHAPED_CANARY
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_flat_call_inside_think_is_rewritten_and_think_stays(self):
        text = (
            '<think>plan <tool_call>{"name":"canary_tool","arguments":{"path":"canary.txt"}}'
            "</tool_call> done</think>"
        )
        expected = "<think>plan " + SHAPED_CANARY + " done</think>"
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_several_blocks_are_each_rewritten(self):
        text = (
            '<tool_call>{"name":"first_tool","arguments":{"n":1}}</tool_call>\n'
            '<tool_call>{"name":"second_tool","arguments":[2]}</tool_call>'
        )
        expected = (
            '<tool_call>{"type":"function","function":{"name":"first_tool",'
            '"arguments":"{\\"n\\":1}"}}</tool_call>\n'
            '<tool_call>{"type":"function","function":{"name":"second_tool",'
            '"arguments":"[2]"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_valid_block_beside_invalid_rewrites_only_the_valid_one(self):
        text = (
            "pre <tool_call>{not json}</tool_call> mid "
            + FLAT_CANARY
            + " post"
        )
        expected = "pre <tool_call>{not json}</tool_call> mid " + SHAPED_CANARY + " post"
        self.assertEqual(normalize_tool_call_text(text), expected)

    def test_text_without_tags_is_unchanged(self):
        text = 'plain {"name":"canary_tool","arguments":{}} prose'
        self.assertEqual(normalize_tool_call_text(text), text)

    def test_empty_string_stays_empty(self):
        self.assertEqual(normalize_tool_call_text(""), "")

    def test_closing_tag_inside_argument_string_still_rewrites(self):
        text = (
            '<tool_call>{"name":"canary_tool","arguments":{"note":"see </tool_call> here"}}'
            "</tool_call>"
        )
        expected = (
            '<tool_call>{"type":"function","function":{"name":"canary_tool",'
            '"arguments":"{\\"note\\":\\"see </tool_call> here\\"}"}}</tool_call>'
        )
        self.assertEqual(normalize_tool_call_text(text), expected)
        calls, _cleaned = extract_tool_calls_from_text(normalize_tool_call_text(text))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "canary_tool")
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"note": "see </tool_call> here"},
        )

    def test_flat_probe_is_invisible_until_normalized_then_extracts_canary(self):
        raw_calls, _cleaned = extract_tool_calls_from_text(FLAT_CANARY)
        self.assertEqual(raw_calls, [])

        calls, _cleaned = extract_tool_calls_from_text(normalize_tool_call_text(FLAT_CANARY))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "canary_tool")
        self.assertEqual(json.loads(calls[0].function.arguments), {"path": "canary.txt"})

    def test_existing_function_name_block_still_yields_one_call(self):
        text = (
            '<tool_call>{"id":"call_1","type":"function","function":{"name":"canary_tool",'
            '"arguments":"{\\"path\\":\\"canary.txt\\"}"}}</tool_call>'
        )
        calls, _cleaned = extract_tool_calls_from_text(normalize_tool_call_text(text))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "canary_tool")
        self.assertEqual(json.loads(calls[0].function.arguments), {"path": "canary.txt"})


if __name__ == "__main__":
    unittest.main()
