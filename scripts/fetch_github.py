#!/usr/bin/env python3
"""Выгрузка коммитов из GitHub REST API в локальный кэш sources/*.jsonl.

Почему это отдельный скрипт, а не стадия dvc. Сеть недетерминирована: тот же
`dvc repro` завтра вернёт другие коммиты, упрётся в лимит запросов или просто
не найдёт интернета. Стадия collect обязана быть воспроизводимой, поэтому
граница проведена так: скрипт ходит в сеть и складывает сырые ответы в файлы,
стадия collect читает только файлы. Кэш версионируется через `dvc add`.

Экономия лимита. Листинг /commits отдаёт сообщение, автора и число родителей,
но не отдаёт список файлов — за ним нужен отдельный запрос на каждый коммит.
Поэтому листинг сохраняется целиком (по нему стадия collect честно считает,
сколько и чего отброшено), а детали запрашиваются только для кандидатов:
не merge, не бот, не revert, не релиз, заголовок по конвенции.

Запуск:
    $env:GITHUB_TOKEN = "..."          # без токена лимит 60 запросов в час
    uv run python scripts/fetch_github.py                  # все репозитории из params
    uv run python scripts/fetch_github.py vuejs/core --max-detail 50
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.commits import is_bot, is_release, is_revert, parse_header  # noqa: E402
from src.config import load_params  # noqa: E402

API_HOST = "api.github.com"
USER_AGENT = "mlops26-hw3-commit-dataset"

# Соединение держим открытым и по одному на поток: запрос к API идёт около
# двух секунд, и почти всё это время — ожидание сети. Три тысячи запросов
# подряд — это часы, те же три тысячи в восемь потоков — минуты.
_local = threading.local()


class RateLimited(RuntimeError):
    """Лимит запросов исчерпан — дальше идти бессмысленно."""


def cache_name(repo: str) -> str:
    """Имя файла кэша для репозитория: owner/name -> owner__name.jsonl."""
    return repo.replace("/", "__") + ".jsonl"


def _send(url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
    """Один GET по живому соединению потока. Порвалось — переподключиться и повторить."""
    last: Exception | None = None
    for _ in range(2):
        conn = getattr(_local, "conn", None)
        if conn is None:
            conn = _local.conn = http.client.HTTPSConnection(API_HOST, timeout=30)
        try:
            conn.request("GET", url, headers=headers)
            response = conn.getresponse()
            body = response.read()  # тело читаем целиком: иначе соединение не переиспользовать
            return response.status, {k.lower(): v for k, v in response.getheaders()}, body
        except (http.client.HTTPException, OSError) as exc:
            last = exc
            conn.close()
            _local.conn = None
    raise RuntimeError(f"{url}: соединение не восстановилось ({last})")


def _follow(location: str) -> str:
    """Путь из заголовка Location. Чужой хост не следуем: соединение на api.github.com."""
    parts = urlsplit(location)
    if parts.netloc and parts.netloc != API_HOST:
        raise RuntimeError(f"переадресация на чужой хост: {location}")
    return parts.path + (f"?{parts.query}" if parts.query else "")


def api_get(
    path: str,
    token: str | None,
    *,
    params: dict[str, object] | None = None,
    retries: int = 3,
) -> tuple[object, dict[str, str]]:
    """GET к API с повторами. Возвращает разобранный JSON и заголовки ответа."""
    url = path
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    attempt = redirects = 0
    while attempt < retries:
        status, response_headers, body = _send(url, headers)
        if status == 200:
            return json.loads(body.decode("utf-8")), response_headers
        if status in (301, 302, 307, 308):
            # Репозиторий переименовали: prisma/prisma -> prisma/orm. API отдаёт
            # 301 с новым адресом, и это нормальная жизнь публичных проектов.
            location = response_headers.get("location")
            if not location or redirects >= 3:
                raise RuntimeError(f"{url}: HTTP {status} без пригодного Location")
            url = _follow(location)
            redirects += 1
            print(f"  переадресация: {url}", flush=True)
            continue
        if status in (403, 429) and response_headers.get("x-ratelimit-remaining") == "0":
            reset = int(response_headers.get("x-ratelimit-reset", "0"))
            raise RateLimited(
                "лимит запросов исчерпан, сброс в "
                + time.strftime("%H:%M:%S", time.localtime(reset))
                + ". Задайте GITHUB_TOKEN — с токеном лимит 5000 запросов в час."
            )
        if status in (403, 429):
            # Вторичный лимит: GitHub просит подождать, а не отказывает совсем.
            time.sleep(int(response_headers.get("retry-after", 2**attempt)))
            attempt += 1
            continue
        if status >= 500:
            time.sleep(2**attempt)
            attempt += 1
            continue
        raise RuntimeError(f"{url}: HTTP {status} — {body[:200].decode('utf-8', 'replace')}")
    raise RuntimeError(f"{url}: не удалось получить ответ за {retries} попытки")


def list_page(repo: str, page: int, per_page: int, token: str | None) -> list[dict]:
    """Одна страница листинга коммитов ветки по умолчанию."""
    payload, _ = api_get(
        f"/repos/{repo}/commits", token, params={"per_page": per_page, "page": page}
    )
    return list(payload)  # type: ignore[arg-type]


def listing_record(repo: str, item: dict) -> dict:
    """Сырая запись листинга: всё, что известно без запроса деталей."""
    author = item.get("author") or {}
    commit = item.get("commit") or {}
    return {
        "repo": repo,
        "sha": item["sha"],
        "message": commit.get("message", ""),
        "date": (commit.get("author") or {}).get("date"),
        "author_login": author.get("login"),
        "author_type": author.get("type"),
        "parents": len(item.get("parents") or []),
        "files": None,
        "files_total": None,
        "stats": None,
    }


def wants_detail(record: dict, types: set[str]) -> bool:
    """Стоит ли тратить запрос на детали этого коммита.

    Фильтр здесь — только экономия лимита, а не отбор в датасет: тот же отбор
    стадия collect делает заново по полному листингу и показывает числами.
    """
    message = record["message"]
    header = parse_header(message)
    if record["parents"] > 1:
        return False
    if is_bot(record["author_login"], record["author_type"]):
        return False
    if header is None or header.type not in types:
        return False
    return not (is_revert(message, header) or is_release(header, message))


def fetch_detail(repo: str, sha: str, token: str | None, max_files: int) -> dict:
    """Список изменённых файлов и статистика по коммиту."""
    payload, _ = api_get(f"/repos/{repo}/commits/{sha}", token)
    files = payload.get("files") or []  # type: ignore[union-attr]
    return {
        "files": [
            {
                "path": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
            }
            for f in files[:max_files]
        ],
        "files_total": len(files),
        "stats": payload.get("stats"),  # type: ignore[union-attr]
    }


def load_cache(path: Path) -> dict[str, dict]:
    """Уже выгруженное. Повторный запуск дотягивает недостающее, а не начинает заново."""
    if not path.exists():
        return {}
    cached: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                cached[row["sha"]] = row
    return cached


def write_cache(path: Path, records: list[dict]) -> None:
    """Переписать кэш целиком: порядок листинга сохраняется."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_repo(
    repo: str,
    cfg: dict,
    types: set[str],
    token: str | None,
    max_detail: int,
    list_pages: int,
    workers: int,
) -> dict:
    """Выгрузить один репозиторий в кэш. Возвращает сводку по репозиторию."""
    out = Path(cfg["dir"]) / cache_name(repo)
    cached = load_cache(out)
    order: list[str] = list(cached)

    listed = 0
    for page in range(1, list_pages + 1):
        items = list_page(repo, page, cfg["per_page"], token)
        if not items:
            break
        for item in items:
            listed += 1
            if item["sha"] in cached:
                continue
            cached[item["sha"]] = listing_record(repo, item)
            order.append(item["sha"])
        print(f"  {repo}: страница {page}, всего в кэше {len(cached)}", flush=True)

    todo = [
        sha
        for sha in order
        if cached[sha].get("files") is None and wants_detail(cached[sha], types)
    ][:max_detail]

    detailed = 0
    limited: RateLimited | None = None
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_detail, repo, sha, token, cfg["max_files_cached"]): sha
                for sha in todo
            }
            for future in as_completed(futures):
                sha = futures[future]
                try:
                    cached[sha].update(future.result())
                except RateLimited as exc:
                    # Лимит общий на все потоки: продолжать бессмысленно.
                    limited = exc
                    for pending in futures:
                        pending.cancel()
                    break
                detailed += 1
                if detailed % 50 == 0:
                    print(f"  {repo}: деталей получено {detailed}/{len(todo)}", flush=True)

    records = [cached[s] for s in order]
    write_cache(out, records)
    with_files = sum(1 for r in records if r.get("files") is not None)
    print(
        f"{repo}: в кэше {len(records)} коммитов, с деталями {with_files} "
        f"(просмотрено {listed}, дотянуто {detailed}) -> {out}"
    )
    if limited is not None:
        raise limited
    return {"repo": repo, "cached": len(records), "with_files": with_files, "fetched": detailed}


def main() -> int:
    params = load_params()
    cfg = params["fetch"]
    types = set(params["collect"]["types"])

    parser = argparse.ArgumentParser(description="выгрузка коммитов в sources/*.jsonl")
    parser.add_argument(
        "repos", nargs="*", help="owner/name; по умолчанию — все из params.fetch.repos"
    )
    parser.add_argument("--max-detail", type=int, default=cfg["max_detail"])
    parser.add_argument("--list-pages", type=int, default=cfg["list_pages"])
    parser.add_argument(
        "--workers",
        type=int,
        default=cfg["workers"],
        help="параллельных запросов за деталями; выше 8 GitHub начинает отвечать 403",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print(
            "GITHUB_TOKEN не задан: лимит анонимных запросов 60 в час, полной\n"
            'выгрузки не выйдет. PowerShell: $env:GITHUB_TOKEN = "<token>"\n',
            file=sys.stderr,
        )

    repos = args.repos or cfg["repos"]
    started = time.perf_counter()
    summary = []
    try:
        for repo in repos:
            summary.append(
                fetch_repo(
                    repo, cfg, types, token, args.max_detail, args.list_pages, args.workers
                )
            )
    except RateLimited as exc:
        print(f"\nостановлено: {exc}", file=sys.stderr)
        print("кэш сохранён, повторный запуск продолжит с того же места", file=sys.stderr)
        return 1

    total = sum(s["with_files"] for s in summary)
    print(f"\nитого коммитов с деталями: {total}, {time.perf_counter() - started:.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
