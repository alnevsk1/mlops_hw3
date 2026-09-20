"""Стадия split: разбиение на train/val/test."""

import json
import random
import time
from pathlib import Path

from src.config import load_params
from src.contamination import is_clean, report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def group_split(sizes: dict[str, int], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Раздать по сплитам ГРУППЫ целиком. Возвращает {группа: имя сплита}.

    Резать по строкам нельзя: у каждого репозитория свой стиль сообщений, и
    коммиты одного проекта в train и в test — это та же утечка, что и дубли,
    только незаметная. На лекции случайный сплит дал 19,5% утечки в test.

    Жадно: группы идут от крупных к мелким, каждая уходит в тот сплит, где
    недобор до целевой доли больше. Точных 80/10/10 при четырёх группах не
    бывает — фактические доли пишутся в metrics/split.json.
    """
    total = sum(sizes.values())
    if len(sizes) < len(ratios):
        raise SystemExit(
            f"групп {len(sizes)}, а сплитов {len(ratios)}: группу нельзя разрезать "
            f"пополам, не создав утечку. Нужен источник с бо́льшим числом групп."
        )

    order = sorted(sizes)
    random.Random(seed).shuffle(order)  # seed решает только ничьи по размеру
    order.sort(key=lambda g: sizes[g], reverse=True)

    assigned: dict[str, str] = {}
    filled = {name: 0 for name in ratios}
    for group in order:
        name = max(ratios, key=lambda n: total * ratios[n] - filled[n])
        assigned[group] = name
        filled[name] += sizes[group]

    # Пустой сплит — не сплит. Отдаём в него самую мелкую группу оттуда,
    # где групп больше одной: так отклонение от целевых долей минимально.
    for name in ratios:
        if filled[name]:
            continue
        donor = max(
            (n for n in ratios if sum(1 for g in assigned.values() if g == n) > 1),
            key=lambda n: filled[n],
        )
        group = min((g for g, n in assigned.items() if n == donor), key=lambda g: sizes[g])
        assigned[group] = name
        filled[donor] -= sizes[group]
        filled[name] += sizes[group]
    return assigned


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    sizes: dict[str, int] = {}
    for ex in examples:
        key = normalize_group(ex.topic)
        sizes[key] = sizes.get(key, 0) + 1

    assigned = group_split(sizes, cfg["ratios"], cfg["seed"])
    buckets: dict[str, list[Example]] = {name: [] for name in cfg["ratios"]}
    for ex in examples:
        buckets[assigned[normalize_group(ex.topic)]].append(ex)

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(sizes),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
        "groups_by_split": {
            name: sorted(g for g, s in assigned.items() if s == name) for name in buckets
        },
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(sizes)}, {metrics['seconds']} с)"
    )

    # Контаминация — падение, а не строчка в логе. Утечка не роняет пайплайн
    # сама по себе: её единственный симптом — метрика, которая приятно удивила.
    if not is_clean(rep):
        for key in ("id_overlap", "text_overlap", "group_overlap", "near_dup_pairs"):
            if rep[key]:
                print(f"  ✗ {key}: {rep[key]}")
        raise SystemExit(
            f"КОНТАМИНАЦИЯ: train и test пересекаются, метрики на test завышены. "
            f"Подробности в {mpath}."
        )


if __name__ == "__main__":
    main()
