"""Register the Cursor ACP model provider with Hermes."""

from __future__ import annotations

from typing import Any

from .client import CursorACPClient
from providers import register_provider
from providers.base import ProviderProfile


class CursorACPProfile(ProviderProfile):
    """External Cursor agent process. The live catalog comes from one ACP session."""

    def create_client(self, **client_kwargs: Any) -> CursorACPClient:
        return CursorACPClient(**client_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str]:
        client = self.create_client()
        try:
            return client.list_models(timeout_seconds=timeout)
        finally:
            client.close()


register_provider(
    CursorACPProfile(
        name="cursor-acp",
        api_mode="chat_completions",
        base_url="acp://cursor",
        auth_type="external_process",
        process_command="agent",
        process_args=("acp",),
        static_models=(),
        fallback_models=(),
    )
)
