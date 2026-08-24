"""
Phase 3: the weekly projection engine. Combines Phase 1 (baselines + matchup
deltas) and Phase 2 (Vegas scaling) into a single projected-points number per
player per week.

Points formula matches Football Machine's existing projectRB/WR/TE/QB exactly
(football_machine.html ~line 4849-4900) - reused, not reinvented:
  RB/WR: (rushYds/10) + (recYds/10) + (TDs*6) + ppr(rec) - fumble penalty
  TE:    same + tePremium*rec
  QB:    (passYds/25) + (passTD*passTdPts) + (rushYds/10) + (rushTD*6) - (ints*2) - fumble penalty
Fumble penalty is omitted here (not part of the agreed efficiency-rate design;
rare enough not to matter for a model that "doesn't need to be sophisticated").

Yards: team's baseline attempt pool -> adjusted by opponent's volume/funnel
delta -> player's share -> player's volume -> player's efficiency baseline
-> adjusted by opponent's position-efficiency delta -> yards.

Touchdowns: NOT derived from the player's own TD rate scaled by a blind final
Vegas multiplier (the old approach had zero connection between "how many
points Vegas expects this team to score" and "how many TDs we project" - a
team implied for 34 points and one implied for 17 got identical treatment
except for an across-the-board multiplier that also touched yardage, which
was already-adjusted volume, i.e. double counting). Instead: Vegas implied
total -> team's own historical TD-points-vs-FG-points split -> implied TD
count -> split rush/pass by the team's own historical ratio -> adjusted by
the opponent's TD-rate matchup delta once at the team-pool level -> allocated
to each player by their existing carry/target share (see
common/vegas.py:compute_implied_team_tds).
"""
from .matchup import (
    compute_player_baseline,
    compute_player_share_baseline,
    compute_team_attempts_baseline,
    compute_position_efficiency_delta,
    compute_team_volume_delta,
)
from .vegas import compute_implied_team_tds

DEFAULT_SCORING = {"format": "PPR", "pass_td_pts": 4, "te_premium": 0}


def _ppr(receptions, scoring):
    fmt = scoring["format"]
    if fmt == "PPR":
        return receptions
    if fmt == "HALF":
        return receptions * 0.5
    return 0.0


def project_player_week(supabase, player_id, position, team, opponent_team, season, week, scoring=None, player_name=None, share_override=None):
    """
    share_override: {"carry_share", "target_share", "pass_attempt_share"} -
    when provided (see common.team_shares.normalize_team_shares), used
    instead of this player's own raw share, since raw per-player shares don't
    sum to 100% across a team on their own (confirmed real bug: could exceed
    it, or leave fringe players with a share more than their real role).
    """
    scoring = scoring or DEFAULT_SCORING

    player_baseline = compute_player_baseline(supabase, player_id, position, season, week, player_name)
    if player_baseline.get("source") == "no_history":
        return {
            "source": "no_history",
            "note": "no career/current-season data and no preseason projections_2026_resolved row found by player_id or name - nothing to project from",
        }

    share_baseline = share_override or compute_player_share_baseline(supabase, player_id, position, team, season, week, player_name)
    team_attempts = compute_team_attempts_baseline(supabase, team, season, week)
    if team_attempts.get("pass_attempts") is None:
        return {"source": "no_team_history", "note": "team has no attempt history to project from"}

    pos_delta = compute_position_efficiency_delta(supabase, opponent_team, position, season, week)
    vol_delta = compute_team_volume_delta(supabase, opponent_team, season, week)
    implied_tds = compute_implied_team_tds(supabase, team, opponent_team, season, week)

    baseline_total_plays = team_attempts["pass_attempts"] + team_attempts["rush_attempts"]
    baseline_rush_share = team_attempts["rush_attempts"] / baseline_total_plays if baseline_total_plays else 0

    adjusted_total_plays = baseline_total_plays * vol_delta["total_plays_ratio"]
    adjusted_rush_share = min(1.0, max(0.0, baseline_rush_share * vol_delta["rush_share_ratio"]))
    adjusted_rush_attempts = adjusted_total_plays * adjusted_rush_share
    adjusted_pass_attempts = adjusted_total_plays * (1 - adjusted_rush_share)

    carry_share = share_baseline.get("carry_share") or 0
    target_share = share_baseline.get("target_share") or 0
    pass_attempt_share = share_baseline.get("pass_attempt_share") or 0

    carries = adjusted_rush_attempts * carry_share
    targets = adjusted_pass_attempts * target_share
    attempts = adjusted_pass_attempts * pass_attempt_share

    efficiencies = {}

    def adj(stat_name):
        base = player_baseline.get(stat_name)
        d = pos_delta.get(stat_name)
        final = None
        if base is not None:
            final = base * d if d is not None else base
        efficiencies[stat_name] = {
            "baseline": round(base, 4) if base is not None else None,
            "opponent_delta": round(d, 3) if d is not None else None,
            "adjusted": round(final, 4) if final is not None else None,
        }
        return final

    rush_ypc = adj("rush_ypc") or 0
    adj("rush_td_rate")  # informational only now - TDs come from implied_tds below, not this rate
    rush_yards = carries * rush_ypc
    # Team's implied rushing TDs, allocated by this player's carry share - not
    # carries x their own historical TD rate (see module docstring).
    rush_tds = implied_tds["rush_tds"] * carry_share

    pts = 0.0
    stat_line = {"carries": round(carries, 2), "rush_yards": round(rush_yards, 1), "rush_tds": round(rush_tds, 3)}

    if position == "QB":
        comp_pct = adj("comp_pct") or 0
        pass_ypa = adj("pass_ypa") or 0
        adj("pass_td_rate")  # informational only - see rush_td_rate note above
        pass_int_rate = adj("pass_int_rate") or 0
        completions = attempts * comp_pct
        pass_yards = attempts * pass_ypa
        # Team's implied passing TDs, allocated by pass-attempt share (~1.0 for the starter).
        pass_tds = implied_tds["pass_tds"] * pass_attempt_share
        ints = attempts * pass_int_rate
        stat_line.update({
            "attempts": round(attempts, 2), "completions": round(completions, 2),
            "pass_yards": round(pass_yards, 1), "pass_tds": round(pass_tds, 3), "ints": round(ints, 3),
        })
        pts = (pass_yards / 25) + (pass_tds * scoring["pass_td_pts"]) + (rush_yards / 10) + (rush_tds * 6) - (ints * 2)
    else:
        catch_rate = adj("catch_rate") or 0
        rec_ypr = adj("rec_ypr") or 0
        adj("rec_td_rate")  # informational only - see rush_td_rate note above
        receptions = targets * catch_rate
        rec_yards = receptions * rec_ypr
        # Team's implied passing TDs, allocated by this player's target share.
        rec_tds = implied_tds["pass_tds"] * target_share
        stat_line.update({
            "targets": round(targets, 2), "receptions": round(receptions, 2),
            "rec_yards": round(rec_yards, 1), "rec_tds": round(rec_tds, 3),
        })
        total_tds = rush_tds + rec_tds
        pts = (rush_yards / 10) + (rec_yards / 10) + (total_tds * 6) + _ppr(receptions, scoring)
        if position == "TE":
            pts += scoring["te_premium"] * receptions

    return {
        "source": player_baseline["source"],
        "games_used": player_baseline["games_used"],
        "implied_team_tds": {k: round(v, 3) for k, v in implied_tds.items()},
        "stat_line": stat_line,
        "efficiencies": efficiencies,
        "shares": {
            "carry_share": round(carry_share, 4) if carry_share else 0,
            "target_share": round(target_share, 4) if target_share else 0,
            "pass_attempt_share": round(pass_attempt_share, 4) if pass_attempt_share else 0,
        },
        "projected_points": round(pts, 2),
    }
