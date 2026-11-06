"""Decision for which model id may be sent on a Cursor ACP pin."""

import unittest

from cursor_acp.model_pin import ModelPinError, resolve_model_pin

PLACEHOLDER = "cursor-acp-unset"


def _resolve(
    requested: str,
    available_ids: list[str],
    *,
    model_entry_disabled: bool = False,
    pin_error: str | None = None,
) -> str:
    return resolve_model_pin(
        requested=requested,
        available_ids=available_ids,
        placeholder_model=PLACEHOLDER,
        model_entry_disabled=model_entry_disabled,
        pin_error=pin_error,
    )


class ResolveModelPinTests(unittest.TestCase):
    def test_returns_advertised_id(self) -> None:
        resolved = resolve_model_pin(
            requested="grok-4.7",
            available_ids=["default", "grok-4.7"],
            placeholder_model=PLACEHOLDER,
            model_entry_disabled=False,
        )

        self.assertEqual(resolved, "grok-4.7")

    def test_returns_stripped_advertised_id(self) -> None:
        resolved = _resolve("\t grok-4.7 \n", ["grok-4.7"])

        self.assertEqual(resolved, "grok-4.7")

    def test_returns_default_when_default_is_advertised(self) -> None:
        resolved = _resolve(" default ", ["grok-4.7", "default"])

        self.assertEqual(resolved, "default")

    def test_returns_auto_when_auto_is_advertised(self) -> None:
        resolved = _resolve(" auto ", ["default", "auto"])

        self.assertEqual(resolved, "auto")

    def test_raises_when_auto_is_absent_and_does_not_send_default(self) -> None:
        with self.assertRaises(ModelPinError) as caught:
            _resolve("auto", ["default", "grok-4.7"])

        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertEqual(caught.exception.available_ids, ("default", "grok-4.7"))
        self.assertIn("default", str(caught.exception))
        self.assertIn("grok-4.7", str(caught.exception))

    def test_raises_when_id_is_not_advertised(self) -> None:
        with self.assertRaises(ModelPinError) as caught:
            _resolve("not-a-cursor-model-hca", ["default", "composer-2.5"])

        self.assertEqual(
            caught.exception.available_ids, ("default", "composer-2.5")
        )
        self.assertIn("default", str(caught.exception))
        self.assertIn("composer-2.5", str(caught.exception))

    def test_raises_when_request_is_blank(self) -> None:
        for requested in ("", "   ", "\n\t"):
            with self.subTest(requested=requested):
                with self.assertRaises(ModelPinError) as caught:
                    _resolve(requested, ["default", ""])

                self.assertEqual(caught.exception.available_ids, ("default", ""))
                self.assertIn("default", str(caught.exception))

    def test_raises_when_request_is_the_placeholder(self) -> None:
        with self.assertRaises(ModelPinError) as caught:
            _resolve(
                f"  {PLACEHOLDER}  ",
                [PLACEHOLDER, "grok-4.7"],
            )

        self.assertEqual(
            caught.exception.available_ids, (PLACEHOLDER, "grok-4.7")
        )
        self.assertIn(PLACEHOLDER, str(caught.exception))
        self.assertIn("grok-4.7", str(caught.exception))

    def test_raises_when_model_entry_is_disabled(self) -> None:
        with self.assertRaises(ModelPinError) as caught:
            _resolve(
                "grok-4.7",
                ["grok-4.7", "default"],
                model_entry_disabled=True,
            )

        self.assertEqual(caught.exception.available_ids, ("grok-4.7", "default"))
        self.assertIn("grok-4.7", str(caught.exception))
        self.assertIn("default", str(caught.exception))

    def test_raises_when_pin_error_is_set(self) -> None:
        with self.assertRaises(ModelPinError) as caught:
            _resolve(
                "grok-4.7",
                ["grok-4.7"],
                pin_error="Invalid model value",
            )

        self.assertEqual(caught.exception.available_ids, ("grok-4.7",))
        self.assertIn("grok-4.7", str(caught.exception))
        self.assertIn("Invalid model value", str(caught.exception))

    def test_raises_when_pin_error_is_empty_string(self) -> None:
        with self.assertRaises(ModelPinError) as caught:
            _resolve("grok-4.7", ["grok-4.7", "default"], pin_error="")

        self.assertEqual(caught.exception.available_ids, ("grok-4.7", "default"))

    def test_error_message_contains_every_available_id(self) -> None:
        available = ["grok-4.7", "composer-2.5", "default"]

        with self.assertRaises(ModelPinError) as caught:
            _resolve("missing-model", available)

        message = str(caught.exception)
        for model_id in available:
            self.assertIn(model_id, message)
        self.assertEqual(caught.exception.available_ids, tuple(available))

    def test_reports_empty_catalog_as_none(self) -> None:
        with self.assertRaises(ModelPinError) as caught:
            _resolve("grok-4.7", [])

        self.assertEqual(caught.exception.available_ids, ())
        self.assertIn("(none)", str(caught.exception))

    def test_matches_catalog_ids_exactly(self) -> None:
        with self.assertRaises(ModelPinError) as caught:
            _resolve("grok-4.7", ["  grok-4.7"])

        self.assertEqual(caught.exception.available_ids, ("  grok-4.7",))
        self.assertIn("  grok-4.7", str(caught.exception))

    def test_error_keeps_catalog_snapshot(self) -> None:
        available = ["grok-4.7", "default"]

        with self.assertRaises(ModelPinError) as caught:
            _resolve("missing-model", available)

        available.append("composer-2.5")
        self.assertEqual(caught.exception.available_ids, ("grok-4.7", "default"))


if __name__ == "__main__":
    unittest.main()
