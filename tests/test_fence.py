"""Private fence for the Cursor ACP child process."""

import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from cursor_acp.fence import Fence

CLI_JSON = (
    b'{"permissions":{"allow":[],"deny":["Read(**)","Read(/**)",'
    b'"Write(**)","Write(/**)","Shell(**)","WebFetch(*)","Mcp(*:*)"]}}'
)
MCP_JSON = b'{"mcpServers":{}}'
FORBIDDEN_ARGV = (
    "--force",
    "--yolo",
    "--sandbox",
    "--approve-mcps",
    "--add-dir",
    "--workspace",
    "--plugin-dir",
    "--worktree",
    "--trust",
    "--model",
    "--print",
    "--resume",
    "--continue",
    "--debug",
    "--api-key",
    "--auth-token",
)
_REPO = Path(__file__).resolve().parents[1]


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _cursor_config_root() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "cursor"
    return (Path.home() / ".config" / "cursor").resolve()


def _author_auth_paths() -> list[Path]:
    paths = [Path.home() / ".config" / "cursor" / "auth.json"]
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        paths.append(Path(xdg).expanduser() / "cursor" / "auth.json")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _rmtree_private(path: Path) -> None:
    """Delete a fence directory that is strictly inside the system temp dir."""
    resolved = path.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    repo = _REPO.resolve()
    if (
        resolved == temp_root
        or resolved == repo
        or not resolved.is_relative_to(temp_root)
        or resolved.is_relative_to(repo)
    ):
        raise AssertionError(f"refusing to delete {resolved}")
    shutil.rmtree(resolved)


class FenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fence: Fence | None = None
        self._extra: list[Path] = []
        self._auth_before = {
            path: path.stat() for path in _author_auth_paths() if path.exists()
        }

    def tearDown(self) -> None:
        if self.fence is not None:
            parent = self.fence.cwd.parent
            try:
                self.fence.close()
            except Exception:
                pass
            if parent.exists():
                resolved = parent.resolve()
                temp_root = Path(tempfile.gettempdir()).resolve()
                if (
                    resolved.is_relative_to(temp_root)
                    and resolved != temp_root
                    and not resolved.is_relative_to(_REPO.resolve())
                ):
                    shutil.rmtree(parent)
        for path in self._extra:
            if not path.exists() and not path.is_symlink():
                continue
            resolved = path.resolve()
            if resolved == _REPO.resolve() or resolved == Path(tempfile.gettempdir()).resolve():
                raise AssertionError(f"refusing to delete {resolved}")
            if path.name.startswith("fence-") and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
        for path, before in self._auth_before.items():
            self.assertTrue(path.exists(), path)
            after = path.stat()
            self.assertEqual(after.st_ino, before.st_ino, path)
            self.assertEqual(after.st_size, before.st_size, path)

    def test_create_makes_private_sibling_directories(self) -> None:
        self.fence = Fence.create()
        parent = self.fence.cwd.parent
        directories = (
            self.fence.cwd,
            self.fence.home,
            self.fence.config_dir,
            self.fence.data_dir,
            self.fence.tmpdir,
        )

        self.assertEqual(_mode(parent), 0o700)
        temp_root = Path(tempfile.gettempdir()).resolve()
        resolved_parent = parent.resolve()
        self.assertTrue(resolved_parent.is_relative_to(temp_root))
        self.assertNotEqual(resolved_parent, temp_root)
        for directory in directories:
            self.assertTrue(directory.is_dir(), directory)
            self.assertEqual(directory.parent, parent)
            self.assertEqual(directory, directory.resolve())
            self.assertEqual(_mode(directory), 0o700)

    def test_create_paths_stay_outside_repo_and_cursor_config(self) -> None:
        self.fence = Fence.create()
        forbidden = (
            _REPO,
            Path.home() / ".cursor",
            _cursor_config_root(),
        )
        for directory in (
            self.fence.cwd,
            self.fence.home,
            self.fence.config_dir,
            self.fence.data_dir,
            self.fence.tmpdir,
        ):
            resolved = directory.resolve()
            for root in forbidden:
                self.assertFalse(
                    resolved.is_relative_to(root.resolve()),
                    f"{resolved} is inside {root}",
                )

    def test_prepare_spawn_writes_cli_json_deny_list(self) -> None:
        self.fence = Fence.create()
        self.fence.prepare_spawn()
        cli_json = self.fence.cwd / ".cursor" / "cli.json"

        self.assertEqual(cli_json.read_bytes(), CLI_JSON)
        self.assertEqual(_mode(cli_json), 0o600)

    def test_prepare_spawn_writes_empty_mcp_json(self) -> None:
        self.fence = Fence.create()
        self.fence.prepare_spawn()
        mcp_json = self.fence.home / ".cursor" / "mcp.json"

        self.assertEqual(mcp_json.read_bytes(), MCP_JSON)
        self.assertEqual(_mode(mcp_json), 0o600)

    def test_prepare_spawn_removes_sandbox_json_and_leaves_cli_config(self) -> None:
        self.fence = Fence.create()
        self.fence.prepare_spawn()
        sandbox = self.fence.cwd / ".cursor" / "sandbox.json"
        sandbox.write_text("residual", encoding="utf-8")
        planted = (
            (self.fence.cwd / ".cursor" / "cli-config.json", "cwd-copy"),
            (self.fence.config_dir / "cli-config.json", "config-copy"),
            (self.fence.home / ".cursor" / "cli-config.json", "home-copy"),
        )
        for path, text in planted:
            path.write_text(text, encoding="utf-8")

        self.fence.prepare_spawn()

        self.assertFalse(sandbox.exists())
        for path, text in planted:
            self.assertEqual(path.read_text(encoding="utf-8"), text)
        self.assertEqual((self.fence.cwd / ".cursor" / "cli.json").read_bytes(), CLI_JSON)

    def test_prepare_spawn_does_not_create_cli_config(self) -> None:
        self.fence = Fence.create()
        self.fence.prepare_spawn()

        self.assertEqual((self.fence.cwd / ".cursor" / "cli.json").read_bytes(), CLI_JSON)
        self.assertFalse((self.fence.cwd / ".cursor" / "cli-config.json").exists())
        self.assertFalse((self.fence.cwd / "cli-config.json").exists())
        self.assertFalse((self.fence.config_dir / "cli-config.json").exists())
        self.assertFalse((self.fence.home / ".cursor" / "cli-config.json").exists())
        self.assertFalse((self.fence.home / "cli-config.json").exists())

    def test_prepare_spawn_recreates_missing_directories_at_the_same_paths(self) -> None:
        self.fence = Fence.create()
        self.fence.prepare_spawn()
        expected = {
            "cwd": self.fence.cwd,
            "home": self.fence.home,
            "config_dir": self.fence.config_dir,
            "data_dir": self.fence.data_dir,
            "tmpdir": self.fence.tmpdir,
        }
        for directory in expected.values():
            _rmtree_private(directory)

        self.fence.prepare_spawn()

        self.assertEqual(self.fence.cwd, expected["cwd"])
        self.assertEqual(self.fence.home, expected["home"])
        self.assertEqual(self.fence.config_dir, expected["config_dir"])
        self.assertEqual(self.fence.data_dir, expected["data_dir"])
        self.assertEqual(self.fence.tmpdir, expected["tmpdir"])
        for directory in expected.values():
            self.assertTrue(directory.is_dir(), directory)
            self.assertEqual(_mode(directory), 0o700)
        self.assertEqual((self.fence.cwd / ".cursor" / "cli.json").read_bytes(), CLI_JSON)
        self.assertEqual((self.fence.home / ".cursor" / "mcp.json").read_bytes(), MCP_JSON)

    def test_prepare_spawn_replaces_cli_json_symlink_without_writing_the_target(
        self,
    ) -> None:
        self.fence = Fence.create()
        self.fence.prepare_spawn()
        outside = Path(tempfile.mkdtemp(prefix="fence-cli-target-"))
        self._extra.append(outside)
        target = outside / "cli.json"
        target.write_text("author-cli", encoding="utf-8")
        link = self.fence.cwd / ".cursor" / "cli.json"
        link.unlink()
        link.symlink_to(target)

        self.fence.prepare_spawn()

        self.assertEqual(target.read_text(encoding="utf-8"), "author-cli")
        self.assertFalse(link.is_symlink())
        self.assertEqual(link.read_bytes(), CLI_JSON)

    def test_prepare_spawn_replaces_cursor_dir_symlink_without_writing_the_target(
        self,
    ) -> None:
        self.fence = Fence.create()
        outside = Path(tempfile.mkdtemp(prefix="fence-cursor-dir-"))
        self._extra.append(outside)
        (outside / "owned.txt").write_text("keep", encoding="utf-8")
        link = self.fence.cwd / ".cursor"
        link.symlink_to(outside, target_is_directory=True)

        self.fence.prepare_spawn()

        self.assertEqual((outside / "owned.txt").read_text(encoding="utf-8"), "keep")
        self.assertFalse((outside / "cli.json").exists())
        self.assertFalse(link.is_symlink())
        self.assertEqual((link / "cli.json").read_bytes(), CLI_JSON)

    def test_child_env_sets_fence_paths_and_keeps_xdg_config_home(self) -> None:
        self.fence = Fence.create()
        env = self.fence.child_env(
            {
                "XDG_CONFIG_HOME": "/hermes/xdg",
                "LANG": "C",
            }
        )

        self.assertEqual(env["HOME"], str(self.fence.home))
        self.assertEqual(env["CURSOR_CONFIG_DIR"], str(self.fence.config_dir))
        self.assertEqual(env["CURSOR_DATA_DIR"], str(self.fence.data_dir))
        self.assertEqual(env["TMPDIR"], str(self.fence.tmpdir))
        self.assertEqual(env["CURSOR_AGENT_DISABLE_DEBUG_LOG"], "1")
        self.assertEqual(env["XDG_CONFIG_HOME"], "/hermes/xdg")
        self.assertEqual(env["LANG"], "C")

    def test_child_env_does_not_set_credentials_or_mutate_process_environment(
        self,
    ) -> None:
        self.fence = Fence.create()
        sentinel = "HERMES_FENCE_ENV_SENTINEL"
        os.environ[sentinel] = "present"
        before_keys = set(os.environ)
        try:
            copied = self.fence.child_env()
            explicit = self.fence.child_env({"LANG": "C"})
            copied[sentinel] = "mutated"
            self.assertEqual(set(os.environ), before_keys)
            self.assertEqual(os.environ[sentinel], "present")
        finally:
            del os.environ[sentinel]

        self.assertEqual(copied["HOME"], str(self.fence.home))
        self.assertEqual(copied["CURSOR_AGENT_DISABLE_DEBUG_LOG"], "1")
        self.assertNotIn("CURSOR_API_KEY", explicit)
        self.assertNotIn("CURSOR_AUTH_TOKEN", explicit)
        self.assertNotIn(sentinel, explicit)
        self.assertEqual(explicit["LANG"], "C")

    def test_command_argv_is_agent_acp_without_bypass_flags(self) -> None:
        self.fence = Fence.create()
        argv = self.fence.command_argv()

        self.assertEqual(argv, ["agent", "acp"])
        for flag in FORBIDDEN_ARGV:
            self.assertNotIn(flag, argv)

    def test_cleanup_turn_removes_session_and_tmpdir_canaries(self) -> None:
        self.fence = Fence.create()
        session_id = "sess-1"
        session_dir = self.fence.config_dir / "acp-sessions" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "store.db-wal").write_text("session-canary", encoding="utf-8")
        (self.fence.tmpdir / "session.log").write_text("tmp-canary", encoding="utf-8")
        other_session = self.fence.config_dir / "acp-sessions" / "other"
        other_session.mkdir()
        (other_session / "keep.txt").write_text("other", encoding="utf-8")
        (self.fence.cwd / "stay.txt").write_text("cwd", encoding="utf-8")
        outside = Path(tempfile.mkdtemp(prefix="fence-outside-"))
        self._extra.append(outside)
        outside_auth = outside / "auth.json"
        outside_auth.write_text("outside-canary", encoding="utf-8")

        self.fence.cleanup_turn(session_id)

        self.assertFalse(session_dir.exists())
        self.assertFalse(self.fence.tmpdir.exists())
        self.assertEqual((other_session / "keep.txt").read_text(encoding="utf-8"), "other")
        self.assertEqual((self.fence.cwd / "stay.txt").read_text(encoding="utf-8"), "cwd")
        self.assertEqual(outside_auth.read_text(encoding="utf-8"), "outside-canary")
        self.assertTrue(self.fence.home.is_dir())
        self.assertTrue(self.fence.config_dir.is_dir())
        self.assertTrue(self.fence.data_dir.is_dir())

    def test_cleanup_turn_refuses_empty_separator_or_dotdot_session_id(self) -> None:
        self.fence = Fence.create()
        session_dir = self.fence.config_dir / "acp-sessions" / "ok"
        session_dir.mkdir(parents=True)
        (session_dir / "store.db-wal").write_text("keep", encoding="utf-8")
        (self.fence.tmpdir / "session.log").write_text("keep", encoding="utf-8")

        for session_id in ("", "a/b", "..", "foo/../bar", "abc..def", "a\\b", "."):
            with self.assertRaises(ValueError):
                self.fence.cleanup_turn(session_id)

        self.assertEqual((session_dir / "store.db-wal").read_text(encoding="utf-8"), "keep")
        self.assertEqual((self.fence.tmpdir / "session.log").read_text(encoding="utf-8"), "keep")

    def test_close_deletes_the_private_parent_and_second_close_is_a_noop(self) -> None:
        self.fence = Fence.create()
        parent = self.fence.cwd.parent
        self.fence.prepare_spawn()

        self.fence.close()

        self.assertFalse(parent.exists())
        self.fence.close()
        self.assertFalse(parent.exists())

    def test_close_refuses_when_the_parent_resolves_outside_system_temp(self) -> None:
        parent = Path(
            tempfile.mkdtemp(prefix="fence-outside-temp-", dir=str(_REPO))
        )
        self._extra.append(parent)
        cwd = parent / "cwd"
        home = parent / "home"
        config_dir = parent / "config"
        data_dir = parent / "data"
        tmpdir = parent / "tmp"
        for directory in (cwd, home, config_dir, data_dir, tmpdir):
            directory.mkdir(mode=0o700)
        fence = Fence(cwd, home, config_dir, data_dir, tmpdir)
        (cwd / "keep.txt").write_text("keep", encoding="utf-8")

        with self.assertRaises(ValueError):
            fence.close()
        with self.assertRaises(ValueError):
            fence.close()

        self.assertEqual((cwd / "keep.txt").read_text(encoding="utf-8"), "keep")
        self.assertTrue(parent.is_dir())



    def test_cleanup_turn_unlinks_tmpdir_symlink_and_leaves_the_target(self) -> None:
        self.fence = Fence.create()
        outside = Path(tempfile.mkdtemp(prefix="fence-tmpdir-target-"))
        self._extra.append(outside)
        canary = outside / "canary.txt"
        canary.write_text("outside-canary", encoding="utf-8")
        self.fence.tmpdir.rmdir()
        self.fence.tmpdir.symlink_to(outside, target_is_directory=True)
        session_dir = self.fence.config_dir / "acp-sessions" / "sess-link"
        session_dir.mkdir(parents=True)
        (session_dir / "store.db-wal").write_text("session-canary", encoding="utf-8")

        raised: BaseException | None = None
        try:
            self.fence.cleanup_turn("sess-link")
        except ValueError as exc:
            raised = exc

        self.assertIsNone(raised)
        self.assertFalse(self.fence.tmpdir.is_symlink())
        self.assertFalse(self.fence.tmpdir.exists())
        self.assertEqual(canary.read_text(encoding="utf-8"), "outside-canary")
        self.assertTrue(outside.is_dir())
        self.assertFalse(session_dir.exists())

    def test_prepare_spawn_after_close_raises_and_leaves_no_tree(self) -> None:
        self.fence = Fence.create()
        parent = self.fence.cwd.parent
        self.fence.close()

        with self.assertRaises(ValueError):
            self.fence.prepare_spawn()

        self.assertFalse(parent.exists())

    def test_cleanup_turn_after_close_raises_and_leaves_no_tree(self) -> None:
        self.fence = Fence.create()
        parent = self.fence.cwd.parent
        self.fence.close()

        with self.assertRaises(ValueError):
            self.fence.cleanup_turn("sess-1")

        self.assertFalse(parent.exists())


if __name__ == "__main__":
    unittest.main()
