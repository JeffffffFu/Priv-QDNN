"""One-off: read result/* attack CSVs and print Python literals for MIA_figure.py."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "result"
NET = "resnet18"
BITS = (10, 8, 6, 4)
MODELS = [
    ("Pytorch-quan", "PYT_PTQ_{b}_{b}"),
    ("AdaRound", "adaround_{b}_{b}"),
    ("QDrop", "qdrop_{b}_{b}"),
    ("PACT", "pact_{b}_{b}"),
    ("HAWQ", "hawq_{b}_{b}"),
    ("BSQ", "bsq_{b}_{b}"),
]
ORIG = "Original_32_32"


def read_kv(p: Path) -> dict[str, float]:
    if not p.is_file():
        return {}
    d: dict[str, float] = {}
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                try:
                    d[row[0].strip()] = float(row[1])
                except ValueError:
                    pass
    return d


def auc_pct(v: float | None) -> float | None:
    if v is None:
        return None
    if v > 1.5:
        return round(float(v), 4)
    return round(float(v) * 100.0, 4)


def grid_compleak(dataset: str, metric_key: str) -> tuple[list[float | None], float | None]:
    vals: list[float | None] = []
    for _, tmpl in MODELS:
        for b in BITS:
            folder = tmpl.format(b=b)
            p = RESULT / "CompLeakNR" / NET / dataset / folder / "attack_performance.csv"
            m = read_kv(p)
            vals.append(auc_pct(m.get(metric_key)))
    p0 = RESULT / "CompLeakNR" / NET / dataset / ORIG / "attack_performance.csv"
    return vals, auc_pct(read_kv(p0).get(metric_key))


def grid_rmia(dataset: str) -> tuple[list[float | None], float | None]:
    vals: list[float | None] = []
    for _, tmpl in MODELS:
        for b in BITS:
            folder = tmpl.format(b=b)
            p = RESULT / "RMIA" / NET / dataset / folder / "attack_performance.csv"
            m = read_kv(p)
            vals.append(auc_pct(m.get("auc_mean")))
    p0 = RESULT / "RMIA" / NET / dataset / ORIG / "attack_performance.csv"
    return vals, auc_pct(read_kv(p0).get("auc_mean"))


def grid_wb(dataset: str, partial: bool) -> tuple[list[float | None], float | None]:
    key = "WhiteBoxMIA_Partial_auc_mean" if partial else "WhiteBoxMIA_auc_mean"
    sub = "WhiteBoxMIA_Partial" if partial else "WhiteBoxMIA"
    vals: list[float | None] = []
    for _, tmpl in MODELS:
        for b in BITS:
            folder = tmpl.format(b=b)
            p = RESULT / sub / NET / dataset / folder / "attack_performance.csv"
            m = read_kv(p)
            vals.append(auc_pct(m.get(key)))
    p0 = RESULT / sub / NET / dataset / ORIG / "attack_performance.csv"
    return vals, auc_pct(read_kv(p0).get(key))


def fmt_values(vals: list[float | None]) -> str:
    lines = []
    for mi in range(len(MODELS)):
        chunk = vals[mi * len(BITS) : (mi + 1) * len(BITS)]
        s = ", ".join(
            f"{x:.4f}" if x is not None and x == x else "float('nan')"
            for x in chunk
        )
        lines.append(f"                    {s},")
    return "\n".join(lines)


def main() -> None:
    attacks: list[tuple[str, str, str | bool | None]] = [
        ("Confidence", "compleak", "Confidence_auc_mean"),
        ("Entropy", "compleak", "Entropy_auc_mean"),
        ("ModifiedEntropy", "compleak", "ModifiedEntropy_auc_mean"),
        ("RF_m1", "compleak", "train_RF_m1_auc_mean"),
        ("SAMIA", "compleak", "SAMIA_auc_mean"),
        ("RMIA", "rmia", None),
        ("WhiteBoxMIA", "wb", False),
        ("WhiteBoxMIA_Partial", "wb_p", True),
    ]
    for ds in ("cifar100", "cinic10"):
        print(f"\n# ----- {ds} -----")
        for name, kind, extra in attacks:
            if kind == "compleak":
                v, b = grid_compleak(ds, str(extra))
            elif kind == "rmia":
                v, b = grid_rmia(ds)
            elif kind == "wb":
                v, b = grid_wb(ds, False)
            else:
                v, b = grid_wb(ds, True)
            nb = f"{b:.4f}" if b is not None and b == b else "float('nan')"
            print(f"        '{name}_{ds}': {{")
            print(f"            'baseline': {nb},")
            print(f"            'title': '{name} ({'CIFAR-100' if ds=='cifar100' else 'CINIC-10'})',")
            print("            'data': {")
            print("                'Value': [")
            print(fmt_values(v))
            print("                ],")
            print("            },")
            print("        },")


if __name__ == "__main__":
    main()
