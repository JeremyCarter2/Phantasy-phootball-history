from __future__ import annotations

from functools import lru_cache

import pandas as pd


def manager_history(team_seasons: pd.DataFrame) -> pd.DataFrame:
    grouped = team_seasons.groupby("Owner Key", as_index=False).agg(
        Manager=("Manager", _latest),
        Seasons=("Season", "nunique"),
        Wins=("Wins", "sum"),
        Losses=("Losses", "sum"),
        Ties=("Ties", "sum"),
        Championships=("Champion", "sum"),
        Playoffs=("Playoff Appearance", "sum"),
        **{
            "Points For": ("Points For", "sum"),
            "Points Against": ("Points Against", "sum"),
            "Best Finish": ("Final Standing", "min"),
        },
    )
    games = grouped["Wins"] + grouped["Losses"] + grouped["Ties"]
    grouped["Win %"] = ((grouped["Wins"] + grouped["Ties"] / 2) / games).fillna(0)
    grouped["Points For"] = grouped["Points For"].round(2)
    grouped["Points Against"] = grouped["Points Against"].round(2)
    return grouped.sort_values(
        ["Championships", "Wins", "Points For"],
        ascending=False,
        ignore_index=True,
    )


def all_play_history(team_history: pd.DataFrame) -> pd.DataFrame:
    regular = team_history[team_history["Season Phase"] == "Regular season"].copy()
    rows = []
    for (season, week), group in regular.groupby(["Season", "Week"]):
        scores = group["Score"].tolist()
        for _, row in group.iterrows():
            wins = sum(row["Score"] > score for score in scores)
            losses = sum(row["Score"] < score for score in scores)
            ties = len(scores) - wins - losses - 1
            rows.append(
                {
                    "Season": season,
                    "Week": week,
                    "Owner Key": row["Owner Key"],
                    "Manager": row["Manager"],
                    "Team": row["Team"],
                    "All-Play Wins": wins,
                    "All-Play Losses": losses,
                    "All-Play Ties": ties,
                }
            )
    weekly = pd.DataFrame(rows)
    if weekly.empty:
        return weekly
    totals = weekly.groupby(["Season", "Owner Key"], as_index=False).agg(
        Manager=("Manager", _latest),
        Team=("Team", _latest),
        **{
            "All-Play Wins": ("All-Play Wins", "sum"),
            "All-Play Losses": ("All-Play Losses", "sum"),
            "All-Play Ties": ("All-Play Ties", "sum"),
        },
    )
    games = (
        totals["All-Play Wins"]
        + totals["All-Play Losses"]
        + totals["All-Play Ties"]
    )
    totals["All-Play %"] = (
        (totals["All-Play Wins"] + totals["All-Play Ties"] / 2) / games
    ).fillna(0)
    return totals


def luck_history(
    team_history: pd.DataFrame,
    team_seasons: pd.DataFrame,
) -> pd.DataFrame:
    all_play = all_play_history(team_history)
    if all_play.empty:
        return all_play
    merged = all_play.merge(
        team_seasons[
            ["Season", "Owner Key", "Wins", "Losses", "Ties"]
        ],
        on=["Season", "Owner Key"],
        how="left",
    )
    actual_games = merged["Wins"] + merged["Losses"] + merged["Ties"]
    merged["Expected Wins"] = (merged["All-Play %"] * actual_games).round(2)
    merged["Luck"] = (merged["Wins"] - merged["Expected Wins"]).round(2)
    return merged.sort_values("Luck", ascending=False, ignore_index=True)


def rivalry_history(team_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Owner A Key",
        "Manager A",
        "Team Names A",
        "Owner B Key",
        "Manager B",
        "Team Names B",
        "Games",
        "Manager A Wins",
        "Manager B Wins",
        "Ties",
        "Manager A Points",
        "Manager B Points",
        "Closest Game",
        "Largest Margin",
    ]
    games = team_history[
        (team_history["Opponent Owner Key"] != "unknown")
        & (team_history["Owner Key"] < team_history["Opponent Owner Key"])
    ].copy()
    rows = []
    for (owner_a, owner_b), group in games.groupby(
        ["Owner Key", "Opponent Owner Key"]
    ):
        a_wins = int((group["Result"] == "W").sum())
        b_wins = int((group["Result"] == "L").sum())
        ties = int((group["Result"] == "T").sum())
        rows.append(
            {
                "Owner A Key": owner_a,
                "Manager A": _latest(group["Manager"]),
                "Team Names A": _unique(group["Team"]),
                "Owner B Key": owner_b,
                "Manager B": _latest(group["Opponent Manager"]),
                "Team Names B": _unique(group["Opponent"]),
                "Games": len(group),
                "Manager A Wins": a_wins,
                "Manager B Wins": b_wins,
                "Ties": ties,
                "Manager A Points": round(group["Score"].sum(), 2),
                "Manager B Points": round(group["Opponent Score"].sum(), 2),
                "Closest Game": round(group["Margin"].abs().min(), 2),
                "Largest Margin": round(group["Margin"].abs().max(), 2),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Games", "Largest Margin"],
        ascending=False,
        ignore_index=True,
    )


def league_records(team_history: pd.DataFrame) -> dict[str, pd.Series]:
    completed = team_history[team_history["Opponent Score"].notna()].copy()
    return {
        "Highest score": team_history.loc[team_history["Score"].idxmax()],
        "Lowest score": team_history.loc[team_history["Score"].idxmin()],
        "Largest win": completed.loc[completed["Margin"].idxmax()],
        "Closest game": completed.loc[completed["Margin"].abs().idxmin()],
    }


def lineup_efficiency(player_history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, week, owner_key), roster in player_history.groupby(
        ["Season", "Week", "Owner Key"]
    ):
        actual = roster[roster["Lineup Status"] == "Starter"]["Points"].sum()
        starter_slots = roster[
            roster["Lineup Status"] == "Starter"
        ]["Lineup Slot"].tolist()
        optimal = _optimal_points(roster, starter_slots)
        bench = roster[roster["Lineup Status"] == "Bench/IR"]["Points"].sum()
        rows.append(
            {
                "Season": season,
                "Week": week,
                "Owner Key": owner_key,
                "Manager": _latest(roster["Manager"]),
                "Team": _latest(roster["Fantasy Team"]),
                "Actual Points": round(actual, 2),
                "Optimal Points": round(optimal, 2),
                "Points Left": round(max(0, optimal - actual), 2),
                "Bench Points": round(bench, 2),
                "Efficiency %": round(actual / optimal, 4) if optimal else 0,
            }
        )
    return pd.DataFrame(rows)


def _optimal_points(roster: pd.DataFrame, slots: list[str]) -> float:
    players = []
    for _, row in roster.iterrows():
        eligible = set(str(row["Eligible Slots"]).split("|"))
        eligible.add(str(row["Position"]))
        players.append((float(row["Points"]), eligible))

    @lru_cache(maxsize=None)
    def solve(slot_index: int, used_mask: int) -> float:
        if slot_index == len(slots):
            return 0.0
        best = solve(slot_index + 1, used_mask)
        slot = slots[slot_index]
        for index, (points, eligible) in enumerate(players):
            if used_mask & (1 << index) or slot not in eligible:
                continue
            best = max(
                best,
                points + solve(slot_index + 1, used_mask | (1 << index)),
            )
        return best

    return solve(0, 0)


def transaction_summary(team_seasons: pd.DataFrame) -> pd.DataFrame:
    grouped = team_seasons.groupby("Owner Key", as_index=False).agg(
        Manager=("Manager", _latest),
        Seasons=("Season", "nunique"),
        Acquisitions=("Acquisitions", "sum"),
        Drops=("Drops", "sum"),
        Trades=("Trades", "sum"),
        **{"FAAB Spent": ("FAAB Spent", "sum")},
    )
    grouped["Moves / Season"] = (
        (grouped["Acquisitions"] + grouped["Trades"]) / grouped["Seasons"]
    ).round(2)
    return grouped.sort_values(
        ["Acquisitions", "Trades"],
        ascending=False,
        ignore_index=True,
    )


def inferred_trade_analysis(
    player_history: pd.DataFrame,
    team_seasons: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "Season",
        "Trade Week",
        "Manager A",
        "Team A",
        "Manager A Received",
        "Manager A Value",
        "Manager B",
        "Team B",
        "Manager B Received",
        "Manager B Value",
        "Winner",
        "Value Margin",
        "Confidence",
    ]
    if player_history.empty:
        return pd.DataFrame(columns=columns)

    ownership = player_history[
        [
            "Season", "Week", "Player ID", "Player", "Owner Key",
            "Manager", "Fantasy Team",
        ]
    ].drop_duplicates(["Season", "Week", "Player ID"])
    moves = []
    for (season, player_id), history in ownership.groupby(
        ["Season", "Player ID"]
    ):
        history = history.sort_values("Week")
        previous = None
        for _, row in history.iterrows():
            if (
                previous is not None
                and row["Week"] == previous["Week"] + 1
                and row["Owner Key"] != previous["Owner Key"]
            ):
                moves.append(
                    {
                        "Season": season,
                        "Trade Week": row["Week"],
                        "Player ID": player_id,
                        "Player": row["Player"],
                        "From Owner": previous["Owner Key"],
                        "From Manager": previous["Manager"],
                        "From Team": previous["Fantasy Team"],
                        "To Owner": row["Owner Key"],
                        "To Manager": row["Manager"],
                        "To Team": row["Fantasy Team"],
                    }
                )
            previous = row

    moves_frame = pd.DataFrame(moves)
    if moves_frame.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (season, week), week_moves in moves_frame.groupby(
        ["Season", "Trade Week"]
    ):
        for owner_a in sorted(week_moves["From Owner"].unique()):
            sent_by_a = week_moves[week_moves["From Owner"] == owner_a]
            for owner_b in sorted(sent_by_a["To Owner"].unique()):
                if owner_a >= owner_b:
                    continue
                a_to_b = week_moves[
                    (week_moves["From Owner"] == owner_a)
                    & (week_moves["To Owner"] == owner_b)
                ]
                b_to_a = week_moves[
                    (week_moves["From Owner"] == owner_b)
                    & (week_moves["To Owner"] == owner_a)
                ]
                if a_to_b.empty or b_to_a.empty:
                    continue

                a_received = b_to_a[["Player ID", "Player"]]
                b_received = a_to_b[["Player ID", "Player"]]
                manager_a = _latest(b_to_a["To Manager"])
                manager_b = _latest(a_to_b["To Manager"])
                team_a = _latest(b_to_a["To Team"])
                team_b = _latest(a_to_b["To Team"])
                a_value = _received_value(
                    player_history, season, week, owner_a, a_received["Player ID"]
                )
                b_value = _received_value(
                    player_history, season, week, owner_b, b_received["Player ID"]
                )
                margin = round(a_value - b_value, 2)
                winner = (
                    manager_a if margin > 0 else manager_b if margin < 0 else "Tie"
                )
                rows.append(
                    {
                        "Season": season,
                        "Trade Week": week,
                        "Manager A": manager_a,
                        "Team A": team_a,
                        "Manager A Received": ", ".join(a_received["Player"]),
                        "Manager A Value": a_value,
                        "Manager B": manager_b,
                        "Team B": team_b,
                        "Manager B Received": ", ".join(b_received["Player"]),
                        "Manager B Value": b_value,
                        "Winner": winner,
                        "Value Margin": abs(margin),
                        "Confidence": "Reciprocal roster moves",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    if team_seasons is not None and not team_seasons.empty:
        for season in result["Season"].unique():
            expected_sides = team_seasons.loc[
                team_seasons["Season"] == season, "Trades"
            ].sum()
            inferred_deals = (result["Season"] == season).sum()
            if expected_sides == inferred_deals * 2:
                result.loc[
                    result["Season"] == season, "Confidence"
                ] = "High - matches ESPN season trade counters"

    return result.sort_values(
        ["Season", "Trade Week"],
        ascending=[False, False],
        ignore_index=True,
    )


def _received_value(
    player_history: pd.DataFrame,
    season: int,
    trade_week: int,
    receiving_owner: str,
    player_ids: pd.Series,
) -> float:
    value = player_history[
        (player_history["Season"] == season)
        & (player_history["Week"] >= trade_week)
        & (player_history["Owner Key"] == receiving_owner)
        & (player_history["Player ID"].isin(player_ids))
    ]["Points"].sum()
    return round(float(value), 2)


def player_roster_history(player_history: pd.DataFrame) -> pd.DataFrame:
    grouped = player_history.groupby(
        ["Player", "Position"],
        as_index=False,
    ).agg(
        Seasons=("Season", "nunique"),
        **{
            "Rostered Weeks": ("Week", "count"),
            "Fantasy Teams": (
                "Fantasy Team",
                _unique,
            ),
            "Managers": ("Manager", _unique),
            "Total Points": ("Points", "sum"),
        },
    )
    grouped["Total Points"] = grouped["Total Points"].round(2)
    return grouped.sort_values(
        ["Rostered Weeks", "Total Points"],
        ascending=False,
        ignore_index=True,
    )


def _latest(values: pd.Series) -> str:
    nonempty = [str(value) for value in values if value]
    return nonempty[-1] if nonempty else "Unknown"


def _unique(values: pd.Series) -> str:
    return ", ".join(dict.fromkeys(str(value) for value in values if value))
