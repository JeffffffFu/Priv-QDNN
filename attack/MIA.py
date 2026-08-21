from attack.CompLeakNR_MIA import CompLeakNR_MIA


def MIA(args):
    """
    Compatibility MIA entrypoint.
    Reuses the black-box confidence attack pipeline so existing
    `--attack_method MIA` experiment scripts keep working.
    """
    args = dict(args)
    # Keep runtime short and deterministic for legacy MIA scripts.
    args.setdefault("compleaknr_attacks", "confidence")
    return CompLeakNR_MIA(args)


def MIA_for_dp(args):
    """
    DP variant compatibility entrypoint.
    Uses the same black-box pipeline as MIA.
    """
    return MIA(args)
