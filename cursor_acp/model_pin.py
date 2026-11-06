"""Decide whether a requested model id may be sent."""


class ModelPinError(RuntimeError):
    """The requested id must not be sent."""

    available_ids: tuple[str, ...]

    def __init__(self, available_ids: tuple[str, ...] | list[str], detail: str) -> None:
        self.available_ids = tuple(available_ids)
        if self.available_ids:
            catalog = ", ".join(self.available_ids)
        else:
            catalog = "(none)"
        super().__init__(f"{detail} Available models: {catalog}")


def resolve_model_pin(
    *,
    requested: str,
    available_ids: list[str],
    placeholder_model: str,
    model_entry_disabled: bool,
    pin_error: str | None = None,
) -> str:
    """Return the id to send, or raise ModelPinError."""
    advertised = tuple(available_ids)
    chosen = requested.strip()
    if (
        chosen
        and chosen != placeholder_model
        and not model_entry_disabled
        and pin_error is None
        and chosen in advertised
    ):
        return chosen

    reasons: list[str] = []
    if not chosen:
        reasons.append("requested model is empty")
    elif chosen == placeholder_model:
        reasons.append(f"requested model {chosen!r} is the placeholder")
    if model_entry_disabled:
        reasons.append("model entry is disabled")
    if chosen not in advertised:
        reasons.append(f"requested model {chosen!r} is not advertised")
    if pin_error is not None:
        reasons.append(pin_error)
    detail = "; ".join(reason for reason in reasons if reason)
    if not detail:
        detail = "model pin refused"
    raise ModelPinError(advertised, f"{detail}.")
