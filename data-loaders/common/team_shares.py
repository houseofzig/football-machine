"""
Team-level share normalization - rescales carry_share/target_share so every
player on a team's rushing pool (RB/WR/QB) and receiving pool (RB/WR/TE)
sums to 100%, matching the existing preseason model's own normalizeBaselineShares
(football_machine.html). Without this, each player's share is computed
independently from their own history alone, with nothing forcing the team's
shares to add up to something real - confirmed causing real problems: the QB
starter bug (every QB with any history got treated as if they played 100% of
snaps) and fringe players (deep bench players with a thin, noisy sample)
holding a share bigger than a full team's worth once summed with the real
starters.

pass_attempt_share (QB) isn't normalized here - compute_player_share_baseline
gives the starter (slot_key ending "_0") 1.0 and every other QB on the team
0.0 (see its is_starter param), so the sum is already exactly 1.0 with
nothing to rescale. get_player_pool no longer filters non-starter QBs out of
the pool entirely (2026-08-28) - they get a real, editable row with 0 default
production instead of not existing at all.
"""
from .matchup import compute_player_share_baseline


def _lock_scale(values, excluded, k, full_sum):
    """ROS only (user, 2026-10-22, "strict lock"): the top `k` players of a pool keep their model
    share; everyone else is scaled so the pool still adds to exactly 100%. Excluded players
    (IR/PUP/Sus) keep their usual full-pool scaling. If the locked players alone already reach
    100% (or nobody else has a share) it falls back to plain even scaling."""
    active = {n: v for n, v in values.items() if n not in excluded}
    total = sum(active.values())
    top = set(sorted((n for n, v in active.items() if v > 0), key=lambda n: -active[n])[:k])
    locked_sum = sum(active[n] for n in top)
    rest = total - locked_sum
    use_lock = rest > 0 and locked_sum < 1
    factor = ((1 - locked_sum) / rest) if use_lock else ((1.0 / total) if total > 0 else 1.0)
    out = {}
    for n, v in values.items():
        if n in excluded:
            out[n] = (v / full_sum) if full_sum > 0 else v
        elif n in top and use_lock:
            out[n] = v
        else:
            out[n] = v * factor
    return out


def normalize_team_shares(supabase, team, players, season, week, saved_ros_weight=None, exclude_ids=None, share_fn=None, starter_ids=None, lock_top=None):
    """
    players: list of {"player_id", "player_name", "position", "slot_key"} for
    every player on this team being projected. Returns {player_name:
    share_dict} with carry_share and target_share rescaled to sum to 1.0
    across the team (only among players who actually carry/target - QB
    carry_share and RB/WR/TE target_share both count toward pools, matching
    how real offenses share touches across positions).

    exclude_ids (weekly only, 2026-10-20): player_ids not playing this week
    (IR/Out/etc). Left out of the sums, so their share goes to the players who
    are playing instead of to nobody - they used to count here and then get
    zeroed afterward, leaving e.g. MIN's active players with ~91% of targets.
    """
    raw = {}
    for p in players:
        # ROS passes the real starting QBs (common/ros_model.starting_qb_ids); the weekly run leaves it None
        is_starter = (p["player_id"] in starter_ids) if starter_ids is not None else (p.get("slot_key") or "").endswith("_0")
        if share_fn is not None:  # ROS model (common/ros_model.py); the weekly run leaves this None
            raw[p["player_name"]] = share_fn(supabase, p["player_id"], p["position"], team, season, week, p["player_name"], is_starter=is_starter)
            continue
        raw[p["player_name"]] = compute_player_share_baseline(
            supabase, p["player_id"], p["position"], team, season, week, p["player_name"],
            is_starter=is_starter, saved_ros_weight=saved_ros_weight,
        )

    excluded = {p["player_name"] for p in players if exclude_ids and p["player_id"] in exclude_ids}
    carry_sum = sum((s.get("carry_share") or 0) for n, s in raw.items() if n not in excluded)
    target_sum = sum((s.get("target_share") or 0) for n, s in raw.items() if n not in excluded)
    # An excluded player's own number: weekly overrides it to 0 afterward (so
    # this value is discarded there); ROS keeps his row as a normal rest-of-
    # season projection, scaled the way it always was (against everyone).
    full_carry_sum = sum((s.get("carry_share") or 0) for s in raw.values())
    full_target_sum = sum((s.get("target_share") or 0) for s in raw.values())

    if lock_top is not None:
        k_target, k_carry = lock_top
        carry_map = _lock_scale({n: (v.get("carry_share") or 0) for n, v in raw.items()}, excluded, k_carry, full_carry_sum)
        target_map = _lock_scale({n: (v.get("target_share") or 0) for n, v in raw.items()}, excluded, k_target, full_target_sum)
        return {n: {"source": v["source"], "carry_share": carry_map[n], "target_share": target_map[n],
                    "pass_attempt_share": v.get("pass_attempt_share")} for n, v in raw.items()}

    normalized = {}
    for name, s in raw.items():
        cs = s.get("carry_share") or 0
        ts = s.get("target_share") or 0
        c_den, t_den = (full_carry_sum, full_target_sum) if name in excluded else (carry_sum, target_sum)
        normalized[name] = {
            "source": s["source"],
            "carry_share": (cs / c_den) if c_den > 0 else cs,
            "target_share": (ts / t_den) if t_den > 0 else ts,
            "pass_attempt_share": s.get("pass_attempt_share"),
        }
    return normalized
