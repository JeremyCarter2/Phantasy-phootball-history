from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from espn_api.football import League


LEAGUE_ID = 682600


@dataclass(frozen=True)
class SeasonResult:
    season: int
    league_name: str
    performances: list[dict[str, Any]]
    team_seasons: list[dict[str, Any]]


@dataclass(frozen=True)
class PlayerSeasonResult:
    season: int
    league_name: str
    performances: list[dict[str, Any]]


def fetch_season(
    season: int,
    espn_s2: str,
    swid: str,
    league_id: int = LEAGUE_ID,
) -> SeasonResult:
    league = League(
        league_id=league_id,
        year=season,
        espn_s2=espn_s2,
        swid=swid,
    )
    performances = extract_performances(
        teams=league.teams,
        season=season,
        regular_season_weeks=league.settings.reg_season_count,
    )
    team_seasons = extract_team_seasons(
        teams=league.teams,
        season=season,
        playoff_team_count=league.settings.playoff_team_count,
    )
    return SeasonResult(
        season=season,
        league_name=league.settings.name,
        performances=performances,
        team_seasons=team_seasons,
    )


def fetch_player_season(
    season: int,
    espn_s2: str,
    swid: str,
    league_id: int = LEAGUE_ID,
) -> PlayerSeasonResult:
    league = League(
        league_id=league_id,
        year=season,
        espn_s2=espn_s2,
        swid=swid,
    )
    regular_season_weeks = league.settings.reg_season_count
    total_weeks = max((len(team.scores) for team in league.teams), default=0)
    rows: list[dict[str, Any]] = []
    player_team_cache: dict[int, int] = {}

    for week in range(1, total_weeks + 1):
        box_scores = league.box_scores(
            week=week,
            player_team_cache=player_team_cache,
        )
        rows.extend(
            extract_player_performances(
                box_scores=box_scores,
                season=season,
                week=week,
                regular_season_weeks=regular_season_weeks,
            )
        )

    return PlayerSeasonResult(
        season=season,
        league_name=league.settings.name,
        performances=rows,
    )


def extract_performances(
    teams: Iterable[Any],
    season: int,
    regular_season_weeks: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for team in teams:
        owner_key, manager = _owner_identity(getattr(team, "owners", []))
        for week_index, score in enumerate(team.scores):
            if score is None:
                continue

            week = week_index + 1
            opponent = _opponent_at(team, week_index)
            opponent_key, opponent_manager = _owner_identity(
                getattr(opponent, "owners", [])
            )
            opponent_score = _score_at(opponent, week_index)
            outcome = _value_at(team.outcomes, week_index, "")

            rows.append(
                {
                    "Season": season,
                    "Week": week,
                    "Owner Key": owner_key,
                    "Manager": manager,
                    "Team": team.team_name,
                    "Score": round(float(score), 2),
                    "Opponent": getattr(opponent, "team_name", "Bye"),
                    "Opponent Owner Key": opponent_key,
                    "Opponent Manager": opponent_manager,
                    "Opponent Score": opponent_score,
                    "Margin": (
                        round(float(score) - opponent_score, 2)
                        if opponent_score is not None
                        else None
                    ),
                    "Result": outcome,
                    "Season Phase": (
                        "Regular season"
                        if week <= regular_season_weeks
                        else "Playoffs"
                    ),
                }
            )

    return rows


def build_dataframe(results: Iterable[SeasonResult]) -> pd.DataFrame:
    rows = [
        performance
        for result in results
        for performance in result.performances
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "Season",
                "Week",
                "Owner Key",
                "Manager",
                "Team",
                "Score",
                "Opponent",
                "Opponent Owner Key",
                "Opponent Manager",
                "Opponent Score",
                "Margin",
                "Result",
                "Season Phase",
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["Score", "Season", "Week"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def extract_player_performances(
    box_scores: Iterable[Any],
    season: int,
    week: int,
    regular_season_weeks: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    for box_score in box_scores:
        for side in ("home", "away"):
            team = getattr(box_score, f"{side}_team", None)
            lineup = getattr(box_score, f"{side}_lineup", [])
            if team is None:
                continue
            owner_key, manager = _owner_identity(getattr(team, "owners", []))

            for player in lineup:
                player_id = int(getattr(player, "playerId", 0))
                unique_key = (season, week, player_id)
                if unique_key in seen:
                    continue
                seen.add(unique_key)

                slot = getattr(player, "slot_position", "")
                breakdown = getattr(player, "breakdown", {}) or {}
                rows.append(
                    {
                        "Season": season,
                        "Week": week,
                        "Player ID": player_id,
                        "Player": player.name,
                        "Position": getattr(player, "position", ""),
                        "NFL Team": getattr(player, "proTeam", ""),
                        "Owner Key": owner_key,
                        "Manager": manager,
                        "Fantasy Team": team.team_name,
                        "Points": round(float(getattr(player, "points", 0)), 2),
                        "Receptions": _stat(breakdown, "receivingReceptions"),
                        "Targets": _stat(breakdown, "receivingTargets"),
                        "Receiving Yards": _stat(breakdown, "receivingYards"),
                        "Receiving TDs": _stat(
                            breakdown, "receivingTouchdowns"
                        ),
                        "Rushing Yards": _stat(breakdown, "rushingYards"),
                        "Rushing TDs": _stat(breakdown, "rushingTouchdowns"),
                        "Passing Yards": _stat(breakdown, "passingYards"),
                        "Passing TDs": _stat(breakdown, "passingTouchdowns"),
                        "Interceptions": _stat(
                            breakdown, "passingInterceptions"
                        ),
                        "Lineup Slot": slot,
                        "Eligible Slots": "|".join(
                            getattr(player, "eligibleSlots", [])
                        ),
                        "Lineup Status": (
                            "Bench/IR" if slot in {"BE", "IR"} else "Starter"
                        ),
                        "Season Phase": (
                            "Regular season"
                            if week <= regular_season_weeks
                            else "Playoffs"
                        ),
                    }
                )

    return rows


def _stat(breakdown: dict[str, Any], key: str) -> float:
    return round(float(breakdown.get(key, 0) or 0), 2)


def build_player_dataframe(
    results: Iterable[PlayerSeasonResult],
) -> pd.DataFrame:
    rows = [
        performance
        for result in results
        for performance in result.performances
    ]
    columns = [
        "Season",
        "Week",
        "Player ID",
        "Player",
        "Position",
        "NFL Team",
        "Owner Key",
        "Manager",
        "Fantasy Team",
        "Points",
        "Receptions",
        "Targets",
        "Receiving Yards",
        "Receiving TDs",
        "Rushing Yards",
        "Rushing TDs",
        "Passing Yards",
        "Passing TDs",
        "Interceptions",
        "Lineup Slot",
        "Eligible Slots",
        "Lineup Status",
        "Season Phase",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(
        ["Points", "Season", "Week"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def extract_team_seasons(
    teams: Iterable[Any],
    season: int,
    playoff_team_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team in teams:
        owner_key, manager = _owner_identity(getattr(team, "owners", []))
        rows.append(
            {
                "Season": season,
                "Owner Key": owner_key,
                "Team": team.team_name,
                "Manager": manager,
                "Wins": team.wins,
                "Losses": team.losses,
                "Ties": team.ties,
                "Points For": round(float(team.points_for), 2),
                "Points Against": round(float(team.points_against), 2),
                "Final Standing": team.final_standing or team.standing,
                "Champion": (team.final_standing or team.standing) == 1,
                "Playoff Appearance": (
                    0 < (team.final_standing or team.standing)
                    <= playoff_team_count
                ),
                "Acquisitions": team.acquisitions,
                "Drops": team.drops,
                "Trades": team.trades,
                "FAAB Spent": team.acquisition_budget_spent,
            }
        )
    return rows


def build_team_seasons_dataframe(
    results: Iterable[SeasonResult],
) -> pd.DataFrame:
    rows = [row for result in results for row in result.team_seasons]
    return pd.DataFrame(rows)


def build_scoring_leaders_dataframe(
    player_history: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Season",
        "Player",
        "Position",
        "NFL Team",
        "Managers",
        "Fantasy Teams",
        "Total Points",
        "Average Points",
        "Best Week",
        "Best Week Points",
        "Weeks Rostered",
    ]
    if player_history.empty:
        return pd.DataFrame(columns=columns)

    leaders = (
        player_history.groupby(
            ["Season", "Player", "Position"],
            dropna=False,
            as_index=False,
        )
        .agg(
            **{
                "NFL Team": ("NFL Team", _latest_nonempty),
                "Managers": ("Manager", _unique_names),
                "Fantasy Teams": ("Fantasy Team", _unique_names),
                "Total Points": ("Points", "sum"),
                "Average Points": ("Points", "mean"),
                "Best Week Points": ("Points", "max"),
                "Weeks Rostered": ("Week", "nunique"),
            }
        )
    )

    best_week_rows = player_history.loc[
        player_history.groupby(
            ["Season", "Player", "Position"],
            dropna=False,
        )["Points"].idxmax(),
        ["Season", "Player", "Position", "Week"],
    ].rename(columns={"Week": "Best Week"})

    leaders = leaders.merge(
        best_week_rows,
        on=["Season", "Player", "Position"],
        how="left",
    )
    for column in ("Total Points", "Average Points", "Best Week Points"):
        leaders[column] = leaders[column].round(2)

    return leaders[columns].sort_values(
        ["Season", "Total Points"],
        ascending=[False, False],
        ignore_index=True,
    )


def _latest_nonempty(values: pd.Series) -> str:
    nonempty = [str(value) for value in values if value]
    return nonempty[-1] if nonempty else ""


def _unique_names(values: pd.Series) -> str:
    return ", ".join(dict.fromkeys(str(value) for value in values if value))


def _manager_name(owners: Iterable[dict[str, Any]]) -> str:
    return _owner_identity(owners)[1]


def _owner_identity(owners: Iterable[dict[str, Any]]) -> tuple[str, str]:
    names = []
    owner_ids = []
    for owner in owners:
        full_name = " ".join(
            part for part in (owner.get("firstName"), owner.get("lastName")) if part
        ).strip()
        name = full_name or owner.get("displayName") or "Unknown"
        names.append(name)
        if owner.get("id"):
            owner_ids.append(str(owner["id"]))

    unique_names = list(dict.fromkeys(names))
    manager = " & ".join(unique_names) or "Unknown"
    if unique_names and unique_names != ["Unknown"]:
        owner_key = "name:" + "|".join(
            sorted(name.casefold().strip() for name in unique_names)
        )
    elif owner_ids:
        owner_key = "id:" + "|".join(sorted(set(owner_ids)))
    else:
        owner_key = "unknown"
    return owner_key, manager


def _opponent_at(team: Any, index: int) -> Any | None:
    opponent = _value_at(team.schedule, index, None)
    if opponent is team:
        return None
    return opponent


def _score_at(team: Any | None, index: int) -> float | None:
    if team is None:
        return None
    score = _value_at(team.scores, index, None)
    return round(float(score), 2) if score is not None else None


def _value_at(values: list[Any], index: int, default: Any) -> Any:
    return values[index] if index < len(values) else default
