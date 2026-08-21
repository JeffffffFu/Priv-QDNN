"""Shared defense bar colors / labels for CIFAR-100 & Location MIA defense figures."""

from __future__ import annotations

# Row order: W/o defense, AR, Drop 15%, Drop 20%, DP ε=1, DP ε=5
DEFENSE_ROW_LABELS: list[str] = [
    "W/o defense",
    "Adversarial regularization",
    r"Drop$_{15\%}$",
    r"Drop$_{20\%}$",
    r"DP $(\epsilon=1)$",
    r"DP $(\epsilon=5)$",
]

DEFENSE_COLORS: list[str] = [
    "#CC5544",  # W/o defense
    "#559955",  # Adversarial regularization
    "#9A8BD4",  # Drop 15%
    "#BEB8DC",  # Drop 20%
    "#4488DD",  # DP ε=1
    "#997766",  # DP ε=5
]
