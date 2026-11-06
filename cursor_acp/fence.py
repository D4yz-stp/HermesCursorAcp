"""Private directories, child environment, and cleanup for a Cursor ACP process."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

_CLI_JSON = (
    b'{"permissions":{"allow":[],"deny":["Read(**)","Read(/**)",'
    b'"Write(**)","Write(/**)","Shell(**)","WebFetch(*)","Mcp(*:*)"]}}'
)
_MCP_JSON = b'{"mcpServers":{}}'
_DIR_MODE = 0o700
_FILE_MODE = 0o600


def _cursor_config_root() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "cursor"
    return (Path.home() / ".config" / "cursor").resolve()


def _forbidden_roots() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve().parents[1],
        (Path.home() / ".cursor").resolve(),
        _cursor_config_root(),
    )


def _reject_session_id(session_id: str) -> None:
    if not isinstance(session_id, str):
        raise ValueError("refusing session id")
    if (
        session_id == ""
        or session_id == "."
        or "/" in session_id
        or "\\" in session_id
        or ".." in session_id
    ):
        raise ValueError("refusing session id")


class Fence:
    """Disposable cwd, home, config, data, and tmp for one child process."""

    def __init__(
        self,
        cwd: Path,
        home: Path,
        config_dir: Path,
        data_dir: Path,
        tmpdir: Path,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.home = Path(home).resolve()
        self.config_dir = Path(config_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.tmpdir = Path(tmpdir).resolve()
        parents = {
            self.cwd.parent,
            self.home.parent,
            self.config_dir.parent,
            self.data_dir.parent,
            self.tmpdir.parent,
        }
        if len(parents) != 1:
            raise ValueError("fence directories must be siblings")
        self._parent = self.cwd.parent
        self._closed = False

    @classmethod
    def create(cls) -> Fence:
        parent = Path(tempfile.mkdtemp(prefix="cursor-acp-fence-")).resolve()
        os.chmod(parent, _DIR_MODE)
        fence = cls(
            parent / "cwd",
            parent / "home",
            parent / "config",
            parent / "data",
            parent / "tmp",
        )
        try:
            for directory in fence._directories():
                fence._mkdir_private(directory)
            fence._reject_forbidden_locations()
        except Exception:
            if parent.is_dir() and not parent.is_symlink():
                shutil.rmtree(parent)
            raise
        return fence

    def prepare_spawn(self) -> None:
        self._require_open()
        self._mkdir_private(self._parent, parents=True)
        for directory in self._directories():
            if directory.parent != self._parent:
                raise ValueError("fence directory is not inside its parent")
            self._mkdir_private(directory)
        self._write_private(self.cwd / ".cursor" / "cli.json", _CLI_JSON)
        self._discard_path(self.cwd / ".cursor" / "sandbox.json")
        self._write_private(self.home / ".cursor" / "mcp.json", _MCP_JSON)

    def child_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env["HOME"] = str(self.home)
        env["CURSOR_CONFIG_DIR"] = str(self.config_dir)
        env["CURSOR_DATA_DIR"] = str(self.data_dir)
        env["TMPDIR"] = str(self.tmpdir)
        env["CURSOR_AGENT_DISABLE_DEBUG_LOG"] = "1"
        return env

    def command_argv(self) -> list[str]:
        return ["agent", "acp"]

    def cleanup_turn(self, session_id: str) -> None:
        self._require_open()
        _reject_session_id(session_id)
        self._delete_session(session_id)
        self._delete_tmpdir()

    def close(self) -> None:
        if self._closed:
            return
        if self._parent.is_symlink():
            raise ValueError(
                "refusing to delete a fence parent outside the system temp directory"
            )
        parent = self._parent.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        repo = Path(__file__).resolve().parents[1].resolve()
        outside_temp = parent == temp_root or not parent.is_relative_to(temp_root)
        inside_repo = parent == repo or parent.is_relative_to(repo)
        if outside_temp or inside_repo:
            raise ValueError(
                "refusing to delete a fence parent outside the system temp directory"
            )
        if parent.exists():
            shutil.rmtree(parent)
        self._closed = True

    def _directories(self) -> tuple[Path, ...]:
        return (self.cwd, self.home, self.config_dir, self.data_dir, self.tmpdir)

    def _reject_forbidden_locations(self) -> None:
        roots = [root.resolve() for root in _forbidden_roots()]
        for directory in self._directories():
            resolved = directory.resolve()
            for root in roots:
                if resolved.is_relative_to(root):
                    raise ValueError(f"{resolved} is inside {root}")

    def _mkdir_private(self, path: Path, *, parents: bool = False) -> None:
        if path.is_symlink():
            path.unlink()
        path.mkdir(mode=_DIR_MODE, parents=parents, exist_ok=True)
        if path.is_symlink():
            raise ValueError(f"refusing symlinked directory {path}")
        os.chmod(path, _DIR_MODE)

    def _require_inside(self, path: Path) -> Path:
        resolved = path.resolve()
        parent = self._parent.resolve()
        if resolved == parent or not resolved.is_relative_to(parent):
            raise ValueError(f"refusing path outside the fence: {resolved}")
        return resolved

    def _write_private(self, path: Path, payload: bytes) -> None:
        self._mkdir_private(path.parent)
        self._require_inside(path.parent)
        if path.is_symlink():
            path.unlink()
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        fd = os.open(path, flags, _FILE_MODE)
        try:
            remaining = payload
            while remaining:
                written = os.write(fd, remaining)
                if written == 0:
                    raise OSError(f"short write to {path}")
                remaining = remaining[written:]
        finally:
            os.close(fd)
        if path.is_symlink():
            raise ValueError(f"refusing to write through symlink {path}")
        os.chmod(path, _FILE_MODE)

    def _discard_path(self, path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        if path.is_symlink():
            self._require_inside(path.parent)
            path.unlink()
            return
        resolved = self._require_inside(path)
        if path.is_dir():
            shutil.rmtree(resolved)
            return
        path.unlink()

    def _delete_session(self, session_id: str) -> None:
        target = self.config_dir / "acp-sessions" / session_id
        if not target.exists() and not target.is_symlink():
            return
        if target.is_symlink():
            self._require_inside(target.parent)
            target.unlink()
            return
        resolved = self._require_inside(target)
        sessions = (self.config_dir / "acp-sessions").resolve()
        if resolved.parent != sessions:
            raise ValueError("refusing to delete session directory")
        shutil.rmtree(resolved)

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("fence is closed")

    def _delete_tmpdir(self) -> None:
        if self.tmpdir.is_symlink():
            if self.tmpdir.parent != self._parent:
                raise ValueError("refusing to delete tmpdir outside the fence")
            self.tmpdir.unlink()
            return
        if not self.tmpdir.exists():
            return
        resolved = self._require_inside(self.tmpdir)
        shutil.rmtree(resolved)
