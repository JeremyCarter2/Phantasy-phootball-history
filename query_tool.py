from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from analytics import (
    lineup_efficiency,
    luck_history,
    manager_history,
    rivalry_history,
)
from espn_history import build_scoring_leaders_dataframe


POSITION_ALIASES = {
    "qb": "QB",
    "quarterback": "QB",
    "quarterbacks": "QB",
    "rb": "RB",
    "running back": "RB",
    "running backs": "RB",
    "wr": "WR",
    "wide receiver": "WR",
    "wide receivers": "WR",
    "te": "TE",
    "tight end": "TE",
    "tight ends": "TE",
    "k": "K",
    "kicker": "K",
    "kickers": "K",
    "dst": "D/ST",
    "d/st": "D/ST",
    "defense": "D/ST",
    "defenses": "D/ST",
}


@dataclass
class QueryResult:
    answer: str
    title: str = "Query result"
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    needs_players: bool = False


def answer_query(
    question: str,
    team_history: pd.DataFrame,
    team_seasons: pd.DataFrame,
    player_history: pd.DataFrame,
) -> QueryResult:
    query = _normalize(question)
    season = _extract_season(query, team_seasons)
    games = _season_filter(team_history, season)
    seasons = _season_filter(team_seasons, season)
    players = _season_filter(player_history, season)

    if not query:
        return _help()

    player_result = _answer_player_query(query, players, player_history.empty)
    if player_result is not None:
        return player_result

    if _contains(query, "highest scoring player", "highest player", "best player game"):
        if player_history.empty:
            return _player_required()
        row = players.loc[players["Points"].idxmax()]
        return QueryResult(
            answer=(
                f"{row['Player']} scored {row['Points']:.2f} points for "
                f"{row['Manager']} in Week {int(row['Week'])}, "
                f"{int(row['Season'])}."
            ),
            title="Highest-scoring player performance",
            table=pd.DataFrame([row]),
        )

    if _contains(query, "scoring leader", "most player points", "player season"):
        if player_history.empty:
            return _player_required()
        leaders = build_scoring_leaders_dataframe(players)
        if leaders.empty:
            return QueryResult("No player scoring data was available.")
        row = leaders.iloc[0]
        return QueryResult(
            answer=(
                f"{row['Player']} led {int(row['Season'])} with "
                f"{row['Total Points']:.2f} total fantasy points."
            ),
            title="Season scoring leader",
            table=leaders.head(20),
        )

    if _contains(query, "lineup", "points left", "manager efficiency"):
        if player_history.empty:
            return _player_required()
        efficiency = lineup_efficiency(players)
        if efficiency.empty:
            return QueryResult("No lineup data was available.")
        if _contains(query, "worst week", "most points left", "most painful"):
            row = efficiency.loc[efficiency["Points Left"].idxmax()]
            return QueryResult(
                answer=(
                    f"{row['Manager']} left {row['Points Left']:.2f} points "
                    f"on the bench in Week {int(row['Week'])}, "
                    f"{int(row['Season'])}."
                ),
                title="Most painful lineup decision",
                table=efficiency.sort_values("Points Left", ascending=False).head(20),
            )
        summary = efficiency.groupby("Manager", as_index=False).agg(
            Actual=("Actual Points", "sum"),
            Optimal=("Optimal Points", "sum"),
            **{"Points Left": ("Points Left", "sum")},
        )
        summary["Efficiency %"] = summary["Actual"] / summary["Optimal"]
        ascending = _contains(query, "worst", "least")
        summary = summary.sort_values("Efficiency %", ascending=ascending)
        row = summary.iloc[0]
        label = "least" if ascending else "most"
        return QueryResult(
            answer=(
                f"{row['Manager']} was the {label} efficient lineup manager "
                f"at {row['Efficiency %']:.1%}."
            ),
            title="Lineup efficiency",
            table=summary,
        )

    if _contains(query, "rivalry", "head to head", "versus", " vs "):
        rivalries = rivalry_history(games)
        managers = _matching_managers(query, team_seasons["Manager"])
        if len(managers) >= 2:
            selected = rivalries[
                (
                    (rivalries["Manager A"] == managers[0])
                    & (rivalries["Manager B"] == managers[1])
                )
                | (
                    (rivalries["Manager A"] == managers[1])
                    & (rivalries["Manager B"] == managers[0])
                )
            ]
            if selected.empty:
                return QueryResult(
                    f"No head-to-head games were found for {managers[0]} and "
                    f"{managers[1]} in the loaded range."
                )
            row = selected.iloc[0]
        elif rivalries.empty:
            return QueryResult("No rivalry data was available.")
        else:
            row = rivalries.sort_values("Games", ascending=False).iloc[0]
        return QueryResult(
            answer=(
                f"{row['Manager A']} and {row['Manager B']} have played "
                f"{int(row['Games'])} times: {row['Manager A']} has "
                f"{int(row['Manager A Wins'])} wins, {row['Manager B']} has "
                f"{int(row['Manager B Wins'])}, with {int(row['Ties'])} ties."
            ),
            title="Head-to-head rivalry",
            table=pd.DataFrame([row]),
        )

    if _contains(query, "luckiest", "most lucky", "unluckiest", "least lucky"):
        luck = luck_history(games, seasons)
        if luck.empty:
            return QueryResult("No schedule-luck data was available.")
        ascending = _contains(query, "unluckiest", "least lucky")
        row = luck.sort_values("Luck", ascending=ascending).iloc[0]
        label = "unluckiest" if ascending else "luckiest"
        return QueryResult(
            answer=(
                f"{row['Manager']} was the {label} in {int(row['Season'])}, "
                f"with {row['Luck']:+.2f} wins versus all-play expectation."
            ),
            title="Schedule luck",
            table=luck.sort_values("Luck", ascending=ascending).head(20),
        )

    if _contains(query, "champion", "championship"):
        if season is not None:
            champion = seasons.sort_values("Final Standing").iloc[0]
            return QueryResult(
                answer=(
                    f"{champion['Manager']} won the {season} championship as "
                    f"{champion['Team']}."
                ),
                title=f"{season} champion",
                table=pd.DataFrame([champion]),
            )
        owners = manager_history(seasons).sort_values(
            ["Championships", "Wins"], ascending=False
        )
        row = owners.iloc[0]
        return QueryResult(
            answer=(
                f"{row['Manager']} has the most championships in the loaded "
                f"archive with {int(row['Championships'])}."
            ),
            title="Championship leaders",
            table=owners,
        )

    if _contains(query, "most wins", "winningest", "best owner", "best manager"):
        owners = manager_history(seasons).sort_values("Wins", ascending=False)
        row = owners.iloc[0]
        return QueryResult(
            answer=f"{row['Manager']} leads with {int(row['Wins'])} wins.",
            title="Win leaders",
            table=owners,
        )

    if _contains(query, "highest score", "highest scoring team", "most points in a game"):
        row = games.loc[games["Score"].idxmax()]
        return QueryResult(
            answer=(
                f"{row['Manager']} scored {row['Score']:.2f} as {row['Team']} "
                f"in Week {int(row['Week'])}, {int(row['Season'])}."
            ),
            title="Highest team score",
            table=pd.DataFrame([row]),
        )

    if _contains(query, "lowest score", "fewest points", "worst score"):
        row = games.loc[games["Score"].idxmin()]
        return QueryResult(
            answer=(
                f"{row['Manager']} scored {row['Score']:.2f} as {row['Team']} "
                f"in Week {int(row['Week'])}, {int(row['Season'])}."
            ),
            title="Lowest team score",
            table=pd.DataFrame([row]),
        )

    if _contains(query, "largest win", "biggest win", "biggest blowout"):
        completed = games[games["Opponent Score"].notna()]
        row = completed.loc[completed["Margin"].idxmax()]
        return QueryResult(
            answer=(
                f"{row['Manager']} beat {row['Opponent Manager']} by "
                f"{row['Margin']:.2f} points in Week {int(row['Week'])}, "
                f"{int(row['Season'])}."
            ),
            title="Largest victory",
            table=pd.DataFrame([row]),
        )

    if _contains(query, "closest game", "closest matchup"):
        completed = games[games["Opponent Score"].notna()]
        row = completed.loc[completed["Margin"].abs().idxmin()]
        return QueryResult(
            answer=(
                f"{row['Manager']} and {row['Opponent Manager']} were separated "
                f"by {abs(row['Margin']):.2f} points in Week "
                f"{int(row['Week'])}, {int(row['Season'])}."
            ),
            title="Closest matchup",
            table=pd.DataFrame([row]),
        )

    managers = _matching_managers(query, team_seasons["Manager"])
    if managers:
        owner = seasons[seasons["Manager"] == managers[0]].sort_values("Season")
        if owner.empty:
            return QueryResult(
                f"No season summary was found for {managers[0]} in that range."
            )
        wins = int(owner["Wins"].sum())
        losses = int(owner["Losses"].sum())
        titles = int(owner["Champion"].sum())
        return QueryResult(
            answer=(
                f"{managers[0]} went {wins}-{losses} with {titles} "
                f"championship{'s' if titles != 1 else ''} in the matching seasons."
            ),
            title=f"{managers[0]} summary",
            table=owner,
        )

    return _help()


def _answer_player_query(
    query: str,
    players: pd.DataFrame,
    player_history_empty: bool,
) -> QueryResult | None:
    position = _extract_position(query)
    player_name = _matching_player(query, players)
    player_terms = (
        "player",
        "scoring leader",
        "fantasy points",
        "best week",
        "highest game",
        "highest scoring",
        "most points",
    )
    is_player_query = (
        position is not None
        or player_name is not None
        or _contains(query, *player_terms)
    )
    if not is_player_query:
        return None
    if player_history_empty:
        return _player_required()
    if players.empty:
        return QueryResult("No player data was available for that season.")

    filtered = players.copy()
    if position is not None:
        filtered = filtered[filtered["Position"] == position]
    if player_name is not None:
        filtered = filtered[
            filtered["Player"].str.casefold() == player_name.casefold()
        ]
    if _contains(query, "starter", "started", "starting"):
        filtered = filtered[filtered["Lineup Status"] == "Starter"]
    elif _contains(query, "bench", "benched"):
        filtered = filtered[filtered["Lineup Status"] == "Bench/IR"]
    if filtered.empty:
        detail = f" at {position}" if position else ""
        return QueryResult(f"No matching player performances were found{detail}.")

    top_n = _extract_top_n(query)
    weekly = _contains(
        query,
        "game",
        "week",
        "performance",
        "single game",
        "single-game",
    )
    lowest = _contains(query, "worst", "lowest", "fewest")

    if player_name is not None and not weekly:
        by_season = build_scoring_leaders_dataframe(filtered)
        by_season = by_season.sort_values(
            "Total Points", ascending=lowest
        ).head(top_n)
        row = by_season.iloc[0]
        return QueryResult(
            answer=(
                f"{row['Player']} scored {row['Total Points']:.2f} total "
                f"fantasy points in {int(row['Season'])}, averaging "
                f"{row['Average Points']:.2f} across "
                f"{int(row['Weeks Rostered'])} weeks."
            ),
            title=f"{player_name} season history",
            table=by_season,
        )

    if weekly:
        board = filtered.sort_values("Points", ascending=lowest).head(top_n)
        row = board.iloc[0]
        position_text = f" {position}" if position else ""
        direction = "lowest" if lowest else "highest"
        return QueryResult(
            answer=(
                f"{row['Player']} had the {direction}{position_text} weekly "
                f"performance with {row['Points']:.2f} points in Week "
                f"{int(row['Week'])}, {int(row['Season'])}."
            ),
            title=f"{direction.title()} weekly player performances",
            table=board,
        )

    leaders = build_scoring_leaders_dataframe(filtered)
    leaders = leaders.sort_values("Total Points", ascending=lowest).head(top_n)
    row = leaders.iloc[0]
    position_text = position or "player"
    direction = "lowest-scoring" if lowest else "best"
    return QueryResult(
        answer=(
            f"{row['Player']} was the {direction} {position_text} in "
            f"{int(row['Season'])} with {row['Total Points']:.2f} total "
            f"fantasy points."
        ),
        title=f"{position_text} season leaders",
        table=leaders,
    )


def _help() -> QueryResult:
    return QueryResult(
        "I could not confidently match that question yet. Try asking about "
        "highest or lowest scores, champions, wins, luck, rivalries, player "
        "performances, scoring leaders, or lineup decisions.",
        title="Try another question",
    )


def _player_required() -> QueryResult:
    return QueryResult(
        "That question needs player box-score data. Turn on **Include player "
        "and lineup questions** in the sidebar and reload the archive.",
        title="Player data required",
        needs_players=True,
    )


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("’", "'").split())


def _contains(query: str, *phrases: str) -> bool:
    return any(phrase in query for phrase in phrases)


def _extract_season(query: str, team_seasons: pd.DataFrame) -> int | None:
    years = {int(year) for year in re.findall(r"\b(20\d{2})\b", query)}
    available = set(team_seasons["Season"].astype(int))
    matches = years & available
    return max(matches) if matches else None


def _season_filter(frame: pd.DataFrame, season: int | None) -> pd.DataFrame:
    if season is None or frame.empty or "Season" not in frame:
        return frame
    return frame[frame["Season"] == season]


def _matching_managers(query: str, managers: pd.Series) -> list[str]:
    names = sorted(set(managers.dropna().astype(str)), key=len, reverse=True)
    return [name for name in names if name.lower() in query]


def _extract_position(query: str) -> str | None:
    for alias in sorted(POSITION_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", query):
            return POSITION_ALIASES[alias]
    return None


def _matching_player(query: str, players: pd.DataFrame) -> str | None:
    if players.empty or "Player" not in players:
        return None
    names = sorted(
        set(players["Player"].dropna().astype(str)),
        key=len,
        reverse=True,
    )
    return next((name for name in names if name.casefold() in query), None)


def _extract_top_n(query: str) -> int:
    match = re.search(r"\btop\s+(\d{1,2})\b", query)
    if not match:
        return 10
    return min(50, max(1, int(match.group(1))))
