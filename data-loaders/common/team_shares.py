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
already hardcodes it to 1.0 for any QB being projected, and only one QB is
ever in the pool per team (see run_mock_projection.get_player_pool), so
there's nothing to normalize across.
"""
from .matchup import compute_player_share_baseline


def normalize_team_shares(supabase, team, players, season, week):
    """
    players: list of {"player_id", "player_name", "position"} for every
    player on this team being projected. Returns {player_name: share_dict}
    with carry_share and target_share rescaled to sum to 1.0 across the team
    (only among players who actually carry/target - QB carry_share and
    RB/WR/TE target_share both count toward pools, matching how real offenses
    share touches across positions).
    """
    raw = {}
    for p in players:
        raw[p["player_name"]] = compute_player_share_baseline(
            supabase, p["player_id"], p["position"], team, season, week, p["player_name"],
        )

    carry_sum = sum((s.get("carry_share") or 0) for s in raw.values())
    target_sum = sum((s.get("target_share") or 0) for s in raw.values())

    normalized = {}
    for name, s in raw.items():
        cs = s.get("carry_share") or 0
        ts = s.get("target_share") or 0
        normalized[name] = {
            "source": s["source"],
            "carry_share": (cs / carry_sum) if carry_sum > 0 else cs,
            "target_share": (ts / target_sum) if target_sum > 0 else ts,
            "pass_attempt_share": s.get("pass_attempt_share"),
        }
    return normalized
