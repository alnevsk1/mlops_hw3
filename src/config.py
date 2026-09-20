"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def source_files(params: dict) -> list[Path]:
    """Файлы-источники для текущей версии датасета.

    Версия живёт в params, а не в аргументах командной строки: иначе
    dvc.lock не запомнит, из чего собран артефакт.
    """
    version = params["collect"]["version"]
    sources = params["collect"]["sources"]
    if version not in sources:
        raise SystemExit(
            f"collect.version = {version!r}, но в collect.sources "
            f"есть только {sorted(sources)}"
        )
    files = [Path(p) for p in sources[version]]
    missing = [f for f in files if not f.exists()]
    if missing:
        # Сообщение должно говорить, что делать, а не печатать
        # FileNotFoundError с чужим абсолютным путём.
        raise SystemExit(
            "стадия collect не нашла кэш источника:\n  "
            + "\n  ".join(str(f) for f in missing)
            + "\n\nКэш собирает scripts/fetch_github.py — он один ходит в сеть.\n"
              "Что сделать:\n"
              '  $env:GITHUB_TOKEN = "<personal access token>"\n'
              "  uv run python scripts/fetch_github.py\n"
              "Без токена лимит GitHub — 60 запросов в час, полной выгрузки не выйдет."
        )
    return files
