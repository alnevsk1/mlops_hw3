"""Стадия collect: кэш коммитов GitHub → data/raw.jsonl.

Источник — выгрузка GitHub REST API из репозиториев, соблюдающих Conventional
Commits (scripts/fetch_github.py кладёт её в sources/*.jsonl). Задача
классификации: по сообщению коммита без префикса и по списку изменённых файлов
предсказать тип (`feat` / `fix` / `docs` / `refactor` / `test` / `chore`) и флаг
breaking change.

Контракт стадии, а не её внутренности, держит остальной пайплайн:
на выходе JSONL со строками {"id", "topic", "messages": [system, user, assistant]}.
`topic` — репозиторий: он же ключ группы для сплита (split.group_key), потому
что у каждого проекта свой стиль сообщений, и коммиты одного репозитория в
train и test завысили бы метрику.

Выгрузка сама по себе сдачей не является: сырой листинг — это сырьё. Стадия
делает из него датасет, и каждое действие видно числом в metrics/collect.json:

  1. выбрасывает то, что является шаблоном, а не языком: merge, revert,
     релизные бампы, коммиты ботов (collect.drop_*);
  2. сверяет заголовок с конвенцией и сужает набор до шести типов
     (collect.types), остальное выбрасывается, а не переносится в обучение;
  3. срезает утечку метки во вход: префикс `type(scope)!:` и строку
     `BREAKING CHANGE:` из тела — иначе задача вырождается в поиск слова;
  4. балансирует классы внутри репозиториев и репозитории между собой
     (collect.max_per_class_per_repo, collect.max_per_repo);
  5. разводит инструкцию на варианты (collect.system_prompts), чтобы модель
     не заучила единственную формулировку.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from src.commits import body_of, is_bot, is_release, is_revert, parse_header, strip_breaking_note
from src.config import load_params, source_files


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбрать вариант инструкции по id примера.

    Именно sha1, а не встроенный hash(): тот солится на каждый запуск процесса,
    и raw.jsonl переставал бы быть воспроизводимым.
    """
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def clip_body(body: str, limit: int) -> str:
    """Обрезать тело сообщения по границе строки, чтобы влезть в фильтр длин."""
    if len(body) <= limit:
        return body
    head = body[:limit]
    cut = head.rfind("\n")
    return (head[:cut] if cut > limit // 2 else head).rstrip() + " …"


def build_user_text(subject: str, body: str, files: list[dict], max_files: int,
                    files_total: int | None) -> str:
    """Вход примера: сообщение без префикса плюс изменённые файлы со статистикой.

    Пути ограничены сверху: коммит, тронувший 300 файлов, иначе забил бы
    полезный текст перечислением и вылетел бы по фильтру длин.
    """
    parts = [subject] if not body else [subject, "", body]
    shown = files[:max_files]
    if shown:
        total = files_total if files_total is not None else len(files)
        header = (
            f"Changed files ({total} total, showing {len(shown)}):"
            if total > len(shown)
            else f"Changed files ({total}):"
        )
        parts += ["", header]
        parts += [
            f"- {f['path']} | {f['status']} | +{f['additions']} -{f['deletions']}" for f in shown
        ]
    return "\n".join(parts).strip()


def read_source(path: Path) -> list[dict]:
    """Прочитать кэш листинга одного репозитория."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: кэш источника не разбирается — {exc.msg}")
    return rows


def main() -> None:  # noqa: C901 — счётчики отбора читаются подряд, дробить хуже
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]

    variants = cfg["system_prompts"]
    if not variants:
        raise SystemExit("collect.system_prompts пуст: инструкцию брать неоткуда")
    types = list(cfg["types"])
    wanted = set(types)

    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    listed = written = 0
    dropped = Counter()
    per_class: Counter[str] = Counter()
    per_repo: Counter[str] = Counter()
    per_repo_class: Counter[tuple[str, str]] = Counter()
    breaking_notes = breaking_true = 0
    prompts_used: set[str] = set()
    seen_ids: set[str] = set()

    with out.open("w", encoding="utf-8") as fh:
        for src in source_files(params):
            for row in read_source(src):
                if written >= cfg["n_rows"]:
                    break
                listed += 1
                repo, sha, message = row["repo"], row["sha"], row["message"]

                # 1. Не язык, а шаблон: merge, бот, revert, релизный бамп.
                header = parse_header(message)
                if cfg["drop_merges"] and row["parents"] > 1:
                    dropped["merge"] += 1
                    continue
                if cfg["drop_bots"] and is_bot(row["author_login"], row["author_type"]):
                    dropped["bot"] += 1
                    continue
                if cfg["drop_reverts"] and is_revert(message, header):
                    dropped["revert"] += 1
                    continue
                if cfg["drop_release"] and is_release(header, message):
                    dropped["release"] += 1
                    continue

                # 2. Конвенция и сужение до шести типов.
                if header is None:
                    dropped["no_prefix"] += 1
                    continue
                if header.type not in wanted:
                    dropped["type_not_wanted"] += 1
                    continue

                # 3. Вход неполон без списка файлов: его нет в листинге,
                #    и за ним нужен отдельный запрос к API.
                if not row.get("files"):
                    dropped["no_file_details"] += 1
                    continue

                example_id = f"{repo}@{sha[:10]}"
                if example_id in seen_ids:
                    dropped["duplicate_id"] += 1
                    continue

                # 4. Балансировка. Потолок класса считается ВНУТРИ репозитория,
                #    а не на весь набор: общий потолок выбирался бы в порядке
                #    файлов, и последнему репозиторию не досталось бы частых
                #    классов вовсе. А репозиторий — это группа сплита, так что
                #    перекос уехал бы прямо в test и в macro-F1.
                if per_repo_class[(repo, header.type)] >= cfg["max_per_class_per_repo"]:
                    dropped["class_cap"] += 1
                    continue
                if per_repo[repo] >= cfg["max_per_repo"]:
                    dropped["repo_cap"] += 1
                    continue

                # 5. Утечка метки во вход: префикс уже отрезан разбором
                #    заголовка, осталась строка BREAKING CHANGE в теле.
                body, notes = strip_breaking_note(body_of(message))
                breaking_notes += notes
                breaking = header.breaking or notes > 0

                user = build_user_text(
                    header.subject,
                    clip_body(body, cfg["max_body_chars"]),
                    row["files"],
                    cfg["max_files"],
                    row.get("files_total"),
                )

                prompt = pick_prompt(example_id, variants)
                prompts_used.add(prompt)
                record = {
                    "id": example_id,
                    "topic": repo,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user},
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {"type": header.type, "breaking": breaking}, ensure_ascii=False
                            ),
                        },
                    ],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_ids.add(example_id)
                per_class[header.type] += 1
                per_repo[repo] += 1
                per_repo_class[(repo, header.type)] += 1
                breaking_true += breaking
                written += 1

    metrics = {
        "version": cfg["version"],
        "files": len(source_files(params)),
        "repos": len(per_repo),
        "commits_listed": listed,
        "rows_written": written,
        "dropped": {
            name: dropped.get(name, 0)
            for name in (
                "merge",
                "bot",
                "revert",
                "release",
                "no_prefix",
                "type_not_wanted",
                "no_file_details",
                "duplicate_id",
                "class_cap",
                "repo_cap",
            )
        },
        "breaking_notes_stripped": breaking_notes,
        "breaking_share": round(breaking_true / written, 4) if written else 0.0,
        "per_class": {t: per_class.get(t, 0) for t in types},
        "per_repo": dict(sorted(per_repo.items())),
        "per_repo_class": {
            repo: {t: per_repo_class.get((repo, t), 0) for t in types}
            for repo in sorted(per_repo)
        },
        "system_prompt_variants": len(prompts_used),
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"collect: версия {cfg['version']}, файлов {metrics['files']}, "
        f"просмотрено {listed}, записано {written} "
        f"(шаблоны -{sum(dropped[k] for k in ('merge', 'bot', 'revert', 'release'))}, "
        f"вне конвенции -{dropped['no_prefix']}, чужие типы -{dropped['type_not_wanted']}, "
        f"без списка файлов -{dropped['no_file_details']}, "
        f"балансировка -{dropped['class_cap'] + dropped['repo_cap']}), "
        f"BREAKING CHANGE вырезан из {breaking_notes} тел, "
        f"вариантов инструкции {len(prompts_used)}, "
        f"{metrics['seconds']} с → {out}"
    )


if __name__ == "__main__":
    main()
