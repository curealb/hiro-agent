"""Local Markdown guidance, isolated enablement, and explicit successful-task drafts.

Only this module's SQL tables are written. Skill text is never execution authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from hiro_agent.bots import BotStore
from hiro_agent.errors import ConfigurationError, PersistenceError
from hiro_agent.kernel_contracts import TaskState, TaskStoreService, utc_now
from hiro_agent.kernel_db import KernelDatabase

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_MAX_BYTES = 1024 * 1024
_GUIDANCE = (
    "Text skills provide task guidance only. They do not grant tool access, "
    "change Bot permissions, or bypass approval."
)


@dataclass(frozen=True)
class SkillDocument:
    id: str
    name: str
    description: str
    content: str
    path: Path
    digest: str


@dataclass(frozen=True)
class SkillSetting:
    bot_id: str
    skill_id: str
    enabled: bool
    updated_at: datetime


@dataclass(frozen=True)
class SkillRecord:
    document: SkillDocument
    enabled: bool
    is_draft: bool


@dataclass(frozen=True)
class SkillDraft:
    bot_id: str
    task_id: str
    document: SkillDocument
    created_at: datetime


def _validate_id(skill_id: str) -> None:
    if not _ID.fullmatch(skill_id):
        raise ConfigurationError("技能 ID 必须是 1-128 位 ASCII 字母数字、下划线或连字符")


def _scalar(value: str, field: str) -> str:
    value = value.strip()
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except ValueError as exc:
            raise ConfigurationError(f"技能 {field} 的双引号文本无效") from exc
        if not isinstance(decoded, str):
            raise ConfigurationError(f"技能 {field} 必须是文本")
        value = decoded
    elif value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise ConfigurationError(f"技能 {field} 的单引号文本无效")
        value = value[1:-1].replace("''", "'")
    elif value.startswith(("[", "{", "&", "*", "!")):
        raise ConfigurationError(f"技能 {field} 仅支持文本或缩进的 |/> 多行文本")
    if not value.strip():
        raise ConfigurationError(f"技能 {field} 不能为空")
    return value.strip()


def _parse(skill_id: str, path: Path, raw: bytes) -> SkillDocument:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"技能文件必须为 UTF-8: {path}") from exc
    if "\x00" in text or not text.strip():
        raise ConfigurationError(f"技能 Markdown 为空或包含 NUL: {path}")
    lines = text.splitlines()
    metadata: dict[str, str] = {}
    if lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i] in ("---", "...")), None)
        if end is None:
            raise ConfigurationError(f"技能 Markdown frontmatter 缺少结束分隔符: {path}")
        index = 1
        seen: set[str] = set()
        while index < end:
            line = lines[index]
            index += 1
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*):\s*(.*)", line)
            if match is None:
                raise ConfigurationError(f"技能 frontmatter 字段格式无效: {path}")
            key, value = match.groups()
            if key in seen:
                raise ConfigurationError(f"技能 frontmatter 字段重复: {key}")
            seen.add(key)
            continuation: list[str] = []
            while index < end and (not lines[index].strip() or lines[index][0].isspace()):
                continuation.append(lines[index])
                index += 1
            # Other metadata, including claimed permissions, is deliberately inert.
            if key not in ("name", "description"):
                continue
            if value in ("|", ">", "|-", ">-", "|+", ">+"):
                separator = "\n" if value.startswith("|") else " "
                value = json.dumps(separator.join(part.strip() for part in continuation).strip())
            elif any(part.strip() for part in continuation):
                raise ConfigurationError(f"技能 {key} 多行文本需要 | 或 > 标记")
            metadata[key] = _scalar(value, key)
        lines = lines[end + 1 :]
    content = "\n".join(lines).strip()
    if not content:
        raise ConfigurationError(f"技能 Markdown 缺少正文: {path}")
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), skill_id)
    paragraph: list[str] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith(("#", "```", "~~~")):
            if paragraph:
                break
            continue
        paragraph.append(line.strip())
    description = metadata.get("description") or " ".join(paragraph)
    if not description:
        raise ConfigurationError(f"技能需要 description 或非空说明段落: {path}")
    name = metadata.get("name", title)
    if not name.strip() or "\n" in name or "\r" in name:
        raise ConfigurationError(f"技能 name 必须是单行非空文本: {path}")
    return SkillDocument(
        skill_id, name, description, content, path, hashlib.sha256(raw).hexdigest()
    )


class SkillStore:
    """A catalog rooted at directory; supported paths are ID.md and ID/SKILL.md."""

    def __init__(
        self,
        database: KernelDatabase,
        bots: BotStore,
        tasks: TaskStoreService,
        directory: Path,
    ) -> None:
        self.database = database
        self.bots = bots
        self.tasks = tasks
        try:
            self.directory = directory.resolve()
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PersistenceError("技能目录无法创建或访问") from exc
        database.migrate(
            "skills",
            1,
            (
                "CREATE TABLE skill_settings (bot_id TEXT NOT NULL, skill_id TEXT NOT NULL, "
                "enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)), updated_at TEXT NOT NULL, "
                "PRIMARY KEY(bot_id, skill_id))",
                "CREATE TABLE skill_drafts (bot_id TEXT NOT NULL, task_id TEXT NOT NULL, "
                "skill_id TEXT NOT NULL UNIQUE, markdown TEXT NOT NULL, created_at TEXT NOT NULL, "
                "published INTEGER NOT NULL DEFAULT 0 CHECK(published IN (0, 1)), "
                "PRIMARY KEY(bot_id, task_id))",
            ),
        )

    @contextmanager
    def _directory_fd(self) -> Iterator[int]:
        try:
            if self.directory.resolve() != self.directory:
                raise ConfigurationError("技能目录不能被替换为符号链接")
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                yield descriptor
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise PersistenceError("技能目录无法读取或写入") from exc

    def _read(self, parts: tuple[str, ...]) -> bytes | None:
        with self._directory_fd() as root:
            parent = root
            try:
                if len(parts) == 2:
                    parent = os.open(
                        parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
                    )
                descriptor = os.open(
                    parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent
                )
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ConfigurationError("技能路径必须是目录边界内的普通文件")
                    raw = stream.read(_MAX_BYTES + 1)
                if len(raw) > _MAX_BYTES:
                    raise ConfigurationError("技能 Markdown 超过 1 MiB 限制")
                return raw
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise ConfigurationError("技能路径不可读取，不能使用符号链接或特殊文件") from exc
            finally:
                if parent != root:
                    os.close(parent)

    def load(self, skill_id: str) -> SkillDocument:
        """Load text only; this grants no Bot access and changes no enablement."""
        _validate_id(skill_id)
        candidates = ((f"{skill_id}.md",), (skill_id, "SKILL.md"))
        found = [(parts, raw) for parts in candidates if (raw := self._read(parts)) is not None]
        if not found:
            raise ConfigurationError(f"技能文件不存在: {skill_id}")
        if len(found) != 1:
            raise ConfigurationError(f"技能 ID 对应多个 Markdown 文件: {skill_id}")
        parts, raw = found[0]
        return _parse(skill_id, self.directory.joinpath(*parts), raw)

    def settings(self, bot_id: str) -> list[SkillSetting]:
        self.bots.get(bot_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM skill_settings WHERE bot_id=? ORDER BY skill_id", (bot_id,)
            ).fetchall()
        return [
            SkillSetting(
                bot_id,
                row["skill_id"],
                bool(row["enabled"]),
                datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def get(self, bot_id: str, skill_id: str) -> SkillRecord:
        settings = {setting.skill_id: setting.enabled for setting in self.settings(bot_id)}
        document = self.load(skill_id)
        with self.database.connect() as connection:
            draft = connection.execute(
                "SELECT 1 FROM skill_drafts WHERE skill_id=?", (skill_id,)
            ).fetchone()
        return SkillRecord(document, settings.get(skill_id, False), draft is not None)

    def list(self, bot_id: str, *, enabled_only: bool = False) -> list[SkillRecord]:
        settings = self.settings(bot_id)
        ids = {setting.skill_id for setting in settings if setting.enabled}
        if not enabled_only:
            with self._directory_fd() as root:
                for name in os.listdir(root):
                    if name.endswith(".md") and _ID.fullmatch(name[:-3]):
                        ids.add(name[:-3])
                    elif _ID.fullmatch(name):
                        mode = os.stat(name, dir_fd=root, follow_symlinks=False).st_mode
                        if stat.S_ISLNK(mode):
                            raise ConfigurationError("技能目录不能包含符号链接技能")
                        if stat.S_ISDIR(mode) and self._read((name, "SKILL.md")) is not None:
                            ids.add(name)
        return [self.get(bot_id, skill_id) for skill_id in sorted(ids)]

    def set_enabled(self, bot_id: str, skill_id: str, enabled: bool) -> SkillSetting:
        self.bots.get(bot_id)
        _validate_id(skill_id)
        if type(enabled) is not bool:
            raise ConfigurationError("技能启停状态必须是布尔值")
        if enabled:
            self.load(skill_id)
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO skill_settings VALUES (?, ?, ?, ?) "
                "ON CONFLICT(bot_id, skill_id) DO UPDATE SET "
                "enabled=excluded.enabled, updated_at=excluded.updated_at",
                (bot_id, skill_id, enabled, now.isoformat()),
            )
        return SkillSetting(bot_id, skill_id, enabled, now)

    def enable(self, bot_id: str, skill_id: str) -> SkillSetting:
        return self.set_enabled(bot_id, skill_id, True)

    def disable(self, bot_id: str, skill_id: str) -> SkillSetting:
        """Disable even a deleted or malformed file, allowing explicit recovery."""
        return self.set_enabled(bot_id, skill_id, False)

    def describe(self, bot_id: str) -> str:
        return self.injection(bot_id)

    def content(self, bot_id: str, skill_id: str) -> str:
        record = self.get(bot_id, skill_id)
        if not record.enabled:
            raise ConfigurationError(f"Bot 尚未启用技能: {skill_id}")
        return record.document.content

    def injection(self, bot_id: str, *, include_content: bool = False) -> str:
        """Return a fresh deterministic prompt fragment; do not modify Session history."""
        records = self.list(bot_id, enabled_only=True)
        if not records:
            return ""
        sections = [_GUIDANCE]
        for record in records:
            document = record.document
            section = f"### Skill {document.id}: {document.name}\n{document.description}"
            if include_content:
                section += f"\n\n{document.content}"
            sections.append(section)
        return "\n\n".join(sections)

    def draft_from_task(self, bot_id: str, task_id: str) -> SkillDraft:
        """Explicitly snapshot a real successful Task; never enable the resulting draft.

        SQL records creation intent before atomic no-clobber file publication. A retry
        repairs a missing file and preserves published user edits. An unpublished,
        conflicting file raises rather than overwriting anything.
        """
        self.bots.get(bot_id)
        task = self.tasks.get(task_id, bot_id=bot_id)
        if task.state != TaskState.SUCCEEDED:
            raise ConfigurationError("只能从真实 succeeded Task 生成技能草稿")
        if not task.output.strip():
            raise ConfigurationError("成功 Task 没有可用于技能草稿的结果")
        identity = json.dumps([bot_id, task_id], ensure_ascii=False, separators=(",", ":"))
        skill_id = "draft-" + hashlib.sha256(identity.encode()).hexdigest()
        longest = max((len(part) for part in re.findall(r"`+", task.text + task.output)), default=0)
        fence = "`" * max(3, longest + 1)
        markdown = (
            "---\n"
            f"name: {json.dumps('Task draft ' + task.id)}\n"
            f"description: {json.dumps('Reusable notes from successful task ' + task.id + '.')}\n"
            "---\n\n# Task-derived skill draft\n\n"
            "Review and edit this draft before explicitly enabling it. "
            + _GUIDANCE
            + f"\n\nSource Bot: {bot_id}\nSource Task: {task.id}\n"
            f"Source Session: {task.session_id}\nTask revision: {task.revision}\n\n"
            f"## Original task\n\n{fence}text\n{task.text}\n{fence}\n\n"
            f"## Successful result\n\n{fence}text\n{task.output}\n{fence}\n"
        )
        if len(markdown.encode()) > _MAX_BYTES:
            raise ConfigurationError("Task 结果生成的技能草稿超过 1 MiB 限制")
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO skill_drafts(bot_id, task_id, skill_id, markdown, created_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(bot_id, task_id) DO NOTHING",
                (bot_id, task_id, skill_id, markdown, utc_now().isoformat()),
            )
            row = connection.execute(
                "SELECT * FROM skill_drafts WHERE bot_id=? AND task_id=?", (bot_id, task_id)
            ).fetchone()
        assert row is not None
        skill_id = row["skill_id"]
        expected = row["markdown"].encode()
        self._publish_draft(skill_id, expected, published=bool(row["published"]))
        document = self.load(skill_id)
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE skill_drafts SET published=1 WHERE bot_id=? AND task_id=?",
                (bot_id, task_id),
            )
        return SkillDraft(bot_id, task_id, document, datetime.fromisoformat(row["created_at"]))

    def _publish_draft(self, skill_id: str, raw: bytes, *, published: bool) -> None:
        if self._read((skill_id, "SKILL.md")) is not None:
            raise ConfigurationError("技能草稿 ID 已存在目录格式技能；保留现有文件")
        existing = self._read((f"{skill_id}.md",))
        if existing is not None:
            if not published and existing != raw:
                raise ConfigurationError("技能草稿文件冲突；保留现有文件，请人工核实")
            return
        temporary = f".draft-{uuid4().hex}.tmp"
        with self._directory_fd() as root:
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=root,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(
                        temporary,
                        f"{skill_id}.md",
                        src_dir_fd=root,
                        dst_dir_fd=root,
                        follow_symlinks=False,
                    )
                    os.fsync(root)
                except FileExistsError:
                    existing = self._read((f"{skill_id}.md",))
                    if existing != raw:
                        raise ConfigurationError("技能草稿文件并发冲突；保留现有文件") from None
            finally:
                try:
                    os.unlink(temporary, dir_fd=root)
                except FileNotFoundError:
                    pass
