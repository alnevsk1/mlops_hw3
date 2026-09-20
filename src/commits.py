"""Разбор Conventional Commits: общий код для стадии collect и для выгрузки.

Здесь живёт всё, что знает про формат `type(scope)!: subject` и про то,
какие коммиты не являются языком: merge, revert, релизные бампы и то,
что написал бот. Один модуль на выгрузку и на стадию collect специально:
если «что такое ботовый коммит» разъедется между ними, счётчики в
metrics/collect.json начнут описывать не тот набор, который поехал в обучение.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Заголовок Conventional Commits. Тип — латиница, scope — в скобках без
# вложенных скобок, `!` перед двоеточием — объявленный breaking change.
HEADER = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^()]*)\))?(?P<bang>!)?:[ \t]+(?P<subject>\S.*)$"
)

# Маркер breaking change в теле сообщения. Это прямая утечка метки во вход:
# строку видно глазами, и классификатор выучил бы её вместо задачи.
BREAKING_NOTE = re.compile(r"^[ \t]*BREAKING[ -]CHANGE[ \t]*:.*$", re.IGNORECASE | re.MULTILINE)

# Релизный бамп: `chore(release): 1.2.3`, `v1.2.3`, `1.2.3` — шаблон, а не язык.
RELEASE_SUBJECT = re.compile(r"^v?\d+\.\d+\.\d+")
REVERT_HEADER = re.compile(r'^revert[ :"]', re.IGNORECASE)

BOT_LOGINS: frozenset[str] = frozenset(
    {
        "dependabot",
        "dependabot-preview",
        "renovate",
        "renovate-bot",
        "greenkeeper",
        "github-actions",
        "semantic-release-bot",
        "angular-robot",
        "vue-bot",
        "prisma-bot",
        "web-flow",
    }
)


@dataclass(frozen=True)
class Header:
    """Разобранный заголовок коммита."""

    type: str
    scope: str | None
    breaking: bool
    subject: str


def parse_header(message: str) -> Header | None:
    """Разобрать первую строку сообщения. None — конвенция не соблюдена."""
    first = message.strip().splitlines()[0] if message.strip() else ""
    match = HEADER.match(first.strip())
    if not match:
        return None
    return Header(
        type=match.group("type").lower(),
        scope=(match.group("scope") or "").strip() or None,
        breaking=bool(match.group("bang")),
        subject=match.group("subject").strip(),
    )


def body_of(message: str) -> str:
    """Тело сообщения — всё после первой строки."""
    lines = message.strip().splitlines()
    return "\n".join(lines[1:]).strip()


def strip_breaking_note(body: str) -> tuple[str, int]:
    """Вырезать из тела маркер BREAKING CHANGE вместе со строкой.

    Возвращает очищенное тело и число вырезанных маркеров. Флаг breaking —
    часть ответа; оставить его текстом во входе значит подсунуть модели ответ.
    """
    cleaned, count = BREAKING_NOTE.subn("", body)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, count


def is_bot(login: str | None, author_type: str | None) -> bool:
    """Коммит написан ботом: шаблон, а не человеческий язык."""
    if (author_type or "").lower() == "bot":
        return True
    name = (login or "").lower()
    return name.endswith("[bot]") or name in BOT_LOGINS


def is_revert(message: str, header: Header | None) -> bool:
    first = message.strip().splitlines()[0] if message.strip() else ""
    if REVERT_HEADER.match(first.strip()):
        return True
    return header is not None and header.type == "revert"


def is_release(header: Header | None, message: str) -> bool:
    """Релизный коммит: `chore(release): x.y.z` и его родня."""
    first = message.strip().splitlines()[0] if message.strip() else ""
    if RELEASE_SUBJECT.match(first.strip()):
        return True
    if header is None:
        return False
    scope = (header.scope or "").lower()
    if scope in {"release", "version", "deps-bump"}:
        return True
    return bool(RELEASE_SUBJECT.match(header.subject))
