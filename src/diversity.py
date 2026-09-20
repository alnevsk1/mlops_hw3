"""Стадия diversity: гейт разнообразия очищенного датасета.

Зачем отдельная стадия. Набор из тридцати строк, сгенерированных по одному
шаблону, проходит и валидацию схемы, и дедупликацию, и проверку контаминации:
формально он безупречен. Обучение на нём сходится быстрее обычного — и модель
выучивает шаблон вместо задачи. Ни одна из предыдущих стадий этого не ловит,
потому что каждая смотрит на строки по отдельности, а вырожденность — свойство
набора целиком.

Гейт, а не отчёт: нарушен порог — стадия падает и печатает, какой именно и
насколько. Пороги живут в params.yaml и меняются осознанно, с обоснованием
в datasheet, а не «чтобы позеленело».
"""

import json
import time
from collections import Counter
from pathlib import Path

from src.config import load_params
from src.schema import iter_examples
from src.stats import spread
from src.textnorm import normalize_group, normalize_text


class DiversityError(ValueError):
    """Набор прошёл все предыдущие стадии, но обучать на нём нечего."""


def measure(path: str, group_key: str) -> dict:
    """Числа, по которым судим о разнообразии. Ничего не решает, только считает."""
    examples = list(iter_examples(path))
    if not examples:
        raise DiversityError(f"{path}: ни одной строки — считать нечего")

    systems = {normalize_text(ex.messages[0].content) for ex in examples}
    groups = Counter(
        normalize_group(getattr(ex, group_key)) if group_key != "topic" else normalize_group(ex.topic)
        for ex in examples
    )
    answers = [ex.assistant for ex in examples]
    lengths = [len(a) for a in answers]
    length_counts = Counter(lengths)
    answer_counts = Counter(normalize_text(a) for a in answers)

    top_group, top_group_n = groups.most_common(1)[0]
    _, top_length_n = length_counts.most_common(1)[0]
    top_answer, top_answer_n = answer_counts.most_common(1)[0]
    duplicate_answers = sum(n - 1 for n in answer_counts.values() if n > 1)

    return {
        "examples": len(examples),
        "system_prompts": len(systems),
        "groups": len(groups),
        "largest_group": top_group,
        "largest_group_share": round(top_group_n / len(examples), 4),
        "answer_len": spread(lengths),
        "same_length_share": round(top_length_n / len(examples), 4),
        "duplicate_answer_share": round(duplicate_answers / len(examples), 4),
        # Оси для задач с законно однотипным ответом: у меток классов разброс
        # длин около единицы по определению, и мерить разнообразие по нему
        # бессмысленно. Мерим его там, где оно должно быть, — в числе классов,
        # в перекосе классов и в разбросе длин ВХОДА.
        "answer_classes": len(answer_counts),
        "largest_answer": top_answer[:60],
        "largest_answer_share": round(top_answer_n / len(examples), 4),
        "user_len": spread([len(ex.user) for ex in examples]),
    }


def violations(stats: dict, cfg: dict) -> list[str]:
    """Список нарушенных порогов. Пустой список — гейт открыт.

    Каждая строка содержит и порог, и фактическое значение: сообщение об ошибке
    должно объяснять, что чинить, а не только что сломалось.
    """
    found: list[str] = []

    if stats["examples"] < cfg["min_examples"]:
        found.append(
            f"мало примеров: {stats['examples']}, нужно ≥ {cfg['min_examples']}"
        )
    if stats["system_prompts"] < cfg["min_system_prompts"]:
        found.append(
            f"системных промптов {stats['system_prompts']}, нужно ≥ {cfg['min_system_prompts']} — "
            f"модель заучит единственную формулировку как константу"
        )
    if stats["groups"] < cfg["min_groups"]:
        found.append(
            f"групп {stats['groups']}, нужно ≥ {cfg['min_groups']}"
        )
    if stats["largest_group_share"] > cfg["max_group_share"]:
        found.append(
            f"крупнейшая группа занимает {stats['largest_group_share']:.1%} "
            f"(порог {cfg['max_group_share']:.0%}): «{stats['largest_group'][:60]}»"
        )
    ratio = stats["answer_len"]["ratio_p90_p10"]
    if ratio < cfg["min_answer_len_ratio"]:
        found.append(
            f"разброс длин ответа p90/p10 = {ratio}, нужно ≥ {cfg['min_answer_len_ratio']} — "
            f"ответы одной длины выдают шаблон"
        )
    if stats["same_length_share"] > cfg["max_same_length_share"]:
        found.append(
            f"{stats['same_length_share']:.1%} ответов имеют одну и ту же длину "
            f"(порог {cfg['max_same_length_share']:.0%})"
        )
    if stats["duplicate_answer_share"] > cfg["max_duplicate_answer_share"]:
        found.append(
            f"{stats['duplicate_answer_share']:.1%} ответов дословно повторяют друг друга "
            f"(порог {cfg['max_duplicate_answer_share']:.0%})"
        )
    if stats["answer_classes"] < cfg["min_answer_classes"]:
        found.append(
            f"различных ответов {stats['answer_classes']}, нужно ≥ {cfg['min_answer_classes']} — "
            f"часть классов в наборе не представлена вовсе"
        )
    if stats["largest_answer_share"] > cfg["max_answer_share"]:
        found.append(
            f"самый частый ответ занимает {stats['largest_answer_share']:.1%} "
            f"(порог {cfg['max_answer_share']:.0%}): «{stats['largest_answer']}» — "
            f"на таком перекосе константный классификатор обыграет модель"
        )
    user_ratio = stats["user_len"]["ratio_p90_p10"]
    if user_ratio < cfg["min_user_len_ratio"]:
        found.append(
            f"разброс длин входа p90/p10 = {user_ratio}, нужно ≥ {cfg['min_user_len_ratio']} — "
            f"входы одного размера выдают шаблон так же, как одинаковые ответы"
        )
    return found


def main() -> None:
    params = load_params()
    cfg = params["diversity"]
    paths = params["paths"]
    started = time.perf_counter()

    stats = measure(paths["clean"], params["split"]["group_key"])
    failed = violations(stats, cfg)

    metrics = {
        "version": params["collect"]["version"],
        **stats,
        "thresholds": dict(cfg),
        "violations": failed,
        "passed": not failed,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_diversity"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failed:
        # Гейт, а не отчёт: стадия, которая сообщает о проблеме и продолжает,
        # не мешает вырожденному набору доехать до обучения. Нарушения печатаются
        # по одному в строку — разбираться придётся с каждым отдельно.
        print(f"diversity: набор не прошёл гейт, нарушено порогов: {len(failed)}")
        for line in failed:
            print(f"  ✗ {line}")
        raise DiversityError(
            f"{paths['clean']}: нарушено порогов разнообразия — {len(failed)}. "
            f"Пороги правятся в params.yaml → diversity, но только с обоснованием "
            f"в docs/datasheet.md."
        )

    print(
        f"diversity: {stats['examples']} строк, {stats['system_prompts']} системных промптов, "
        f"{stats['groups']} групп (крупнейшая {stats['largest_group_share']:.1%}), "
        f"разброс длин p90/p10 = {stats['answer_len']['ratio_p90_p10']}, "
        f"{metrics['seconds']} с"
    )


if __name__ == "__main__":
    main()
