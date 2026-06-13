from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from analytics import (
    inferred_trade_analysis,
    lineup_efficiency,
    luck_history,
    manager_history,
    rivalry_history,
)
from espn_history import build_scoring_leaders_dataframe
from draft_history import all_draft_value


POSITION_ALIASES = {
    "qb": "QB",
    "qbs": "QB",
    "quarterback": "QB",
    "quarterbacks": "QB",
    "rb": "RB",
    "rbs": "RB",
    "running back": "RB",
    "running backs": "RB",
    "wr": "WR",
    "wrs": "WR",
    "wide receiver": "WR",
    "wide receivers": "WR",
    "te": "TE",
    "tes": "TE",
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

STAT_ALIASES = {
    "catches": ("Receptions", "receptions"),
    "catch": ("Receptions", "receptions"),
    "receptions": ("Receptions", "receptions"),
    "reception": ("Receptions", "receptions"),
    "targets": ("Targets", "targets"),
    "target": ("Targets", "targets"),
    "receiving yards": ("Receiving Yards", "receiving yards"),
    "receiving touchdowns": ("Receiving TDs", "receiving touchdowns"),
    "receiving tds": ("Receiving TDs", "receiving touchdowns"),
    "rushing yards": ("Rushing Yards", "rushing yards"),
    "rushing touchdowns": ("Rushing TDs", "rushing touchdowns"),
    "rushing tds": ("Rushing TDs", "rushing touchdowns"),
    "passing yards": ("Passing Yards", "passing yards"),
    "passing touchdowns": ("Passing TDs", "passing touchdowns"),
    "passing tds": ("Passing TDs", "passing touchdowns"),
    "interceptions": ("Interceptions", "interceptions"),
}


@dataclass
class QueryResult:
    answer: str
    title: str = "Query result"
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    needs_players: bool = False
    interpretation: str = ""
    suggestions: tuple[str, ...] = ()


def answer_query(
    question: str,
    team_history: pd.DataFrame,
    team_seasons: pd.DataFrame,
    player_history: pd.DataFrame,
    draft_history: pd.DataFrame | None = None,
) -> QueryResult:
    query = _normalize(question)
    season = _extract_season(query, team_seasons)
    games = _season_filter(team_history, season)
    seasons = _season_filter(team_seasons, season)
    players = _season_filter(player_history, season)

    if not query:
        return _help()

    journey_result = _answer_player_journey_query(
        query,
        player_history,
        draft_history if draft_history is not None else pd.DataFrame(),
        team_seasons,
    )
    if journey_result is not None:
        return journey_result

    draft_result = _answer_draft_query(
        query,
        season,
        draft_history if draft_history is not None else pd.DataFrame(),
        player_history,
        team_seasons,
    )
    if draft_result is not None:
        return draft_result

    trade_result = _answer_trade_query(
        query,
        season,
        player_history,
        team_seasons,
    )
    if trade_result is not None:
        return trade_result

    comparison_result = _answer_player_comparison_query(
        query,
        players,
        player_history.empty,
        season,
    )
    if comparison_result is not None:
        return comparison_result

    ownership_result = _answer_player_ownership_query(
        query,
        players,
        player_history.empty,
        season,
    )
    if ownership_result is not None:
        return ownership_result

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
                    f"{champion['Manager']} won the {season} championship."
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
                f"{row['Manager']} scored {row['Score']:.2f} in Week "
                f"{int(row['Week'])}, {int(row['Season'])}."
            ),
            title="Highest team score",
            table=pd.DataFrame([row]),
        )

    if _contains(query, "lowest score", "fewest points", "worst score"):
        row = games.loc[games["Score"].idxmin()]
        return QueryResult(
            answer=(
                f"{row['Manager']} scored {row['Score']:.2f} in Week "
                f"{int(row['Week'])}, {int(row['Season'])}."
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


def _answer_player_journey_query(
    query: str,
    player_history: pd.DataFrame,
    drafts: pd.DataFrame,
    team_seasons: pd.DataFrame,
) -> QueryResult | None:
    asks_for_journey = _contains(
        query,
        "full history",
        "complete history",
        "player history",
        "journey",
        "timeline",
        "draft and trade",
        "drafted and traded",
        "draft history and trade",
    )
    has_mixed_intents = (
        _contains(query, "draft", "drafted", "auction")
        and _is_trade_query(query)
    )
    if not asks_for_journey and not has_mixed_intents:
        return None

    player_name = _matching_player_across_sources(query, player_history, drafts)
    if player_name is None:
        return QueryResult(
            "I understood this as a player timeline question, but I could not "
            "identify the player. Include the player's full name.",
            title="Player timeline needs a name",
            interpretation="Player draft and trade timeline",
        )
    if player_history.empty:
        return _player_required(
            "A complete player timeline needs weekly roster data."
        )

    target = _normalize_player_name(player_name)
    events: list[dict[str, object]] = []
    if not drafts.empty:
        purchases = drafts[
            drafts["Player"].map(_normalize_player_name) == target
        ]
        for _, row in purchases.iterrows():
            events.append(
                {
                    "Season": int(row["Season"]),
                    "Week": 0,
                    "Event": "Drafted",
                    "Owner": row["Owner"],
                    "Other Owner": "",
                    "Details": f"${int(row['Price'])}",
                }
            )

    trades = inferred_trade_analysis(player_history, team_seasons)
    for _, row in trades.iterrows():
        if not _player_in_trade(row, player_name):
            continue
        a_received = _trade_side_contains(row["Manager A Received"], target)
        events.append(
            {
                "Season": int(row["Season"]),
                "Week": int(row["Trade Week"]),
                "Event": "Inferred trade",
                "Owner": row["Manager A"] if a_received else row["Manager B"],
                "Other Owner": (
                    row["Manager B"] if a_received else row["Manager A"]
                ),
                "Details": (
                    row["Manager A Received"]
                    if a_received
                    else row["Manager B Received"]
                ),
            }
        )

    timeline = pd.DataFrame(events)
    if timeline.empty:
        return QueryResult(
            f"No draft purchases or inferred trades were found for {player_name}.",
            title=f"{player_name} timeline",
            interpretation=f"Complete archive timeline for {player_name}",
        )
    timeline = timeline.sort_values(["Season", "Week"], ignore_index=True)
    draft_count = int((timeline["Event"] == "Drafted").sum())
    trade_count = int((timeline["Event"] == "Inferred trade").sum())
    draft_noun = "time" if draft_count == 1 else "times"
    trade_noun = "trade" if trade_count == 1 else "trades"
    return QueryResult(
        f"{player_name} was drafted {draft_count} {draft_noun} and appears in "
        f"{trade_count} inferred {trade_noun} in the loaded archive.",
        title=f"{player_name} archive timeline",
        table=timeline,
        interpretation=f"Draft plus inferred trade history for {player_name}",
        suggestions=(
            f"Who rostered {player_name} the longest?",
            f"How many fantasy points did {player_name} score each season?",
        ),
    )


def _answer_player_comparison_query(
    query: str,
    players: pd.DataFrame,
    player_history_empty: bool,
    season: int | None,
) -> QueryResult | None:
    if not _contains(query, "compare", " vs ", " versus ", "better"):
        return None
    matches = _matching_players(query, players)
    if len(matches) < 2:
        return None
    if player_history_empty:
        return _player_required("That player comparison needs box-score data.")

    selected = players[
        players["Player"].map(_normalize_player_name).isin(
            {_normalize_player_name(name) for name in matches[:2]}
        )
    ]
    leaders = build_scoring_leaders_dataframe(selected).sort_values(
        ["Season", "Total Points"],
        ascending=[False, False],
    )
    if leaders.empty:
        return QueryResult("No matching player performances were found.")
    totals = (
        leaders.groupby("Player", as_index=False)
        .agg(
            Seasons=("Season", "nunique"),
            **{
                "Total Points": ("Total Points", "sum"),
                "Average Weekly Points": ("Average Points", "mean"),
                "Weeks Rostered": ("Weeks Rostered", "sum"),
            },
        )
        .sort_values("Total Points", ascending=False)
    )
    totals["Total Points"] = totals["Total Points"].round(2)
    totals["Average Weekly Points"] = totals["Average Weekly Points"].round(2)
    leader = totals.iloc[0]
    scope = f"in {season}" if season is not None else "in the loaded archive"
    return QueryResult(
        f"{leader['Player']} leads the comparison {scope} with "
        f"{leader['Total Points']:.2f} total fantasy points.",
        title=f"{matches[0]} vs. {matches[1]}",
        table=leaders if season is None else totals,
        interpretation=(
            f"Fantasy-point comparison: {matches[0]} versus {matches[1]}"
        ),
        suggestions=(
            f"What was {matches[0]}'s best week?",
            f"What was {matches[1]}'s best week?",
        ),
    )


def _answer_player_ownership_query(
    query: str,
    players: pd.DataFrame,
    player_history_empty: bool,
    season: int | None,
) -> QueryResult | None:
    if not _contains(
        query,
        "who owned",
        "who rostered",
        "who had",
        "owners had",
        "which owners",
        "how many owners",
        "owner history",
        "rostered the longest",
        "owned the longest",
        "played for",
    ):
        return None
    player_name = _matching_player(query, players)
    if player_name is None:
        return None
    if player_history_empty:
        return _player_required("That ownership question needs roster data.")

    selected = players[
        players["Player"].map(_normalize_player_name)
        == _normalize_player_name(player_name)
    ]
    ownership = (
        selected.groupby(["Season", "Manager"], as_index=False)
        .agg(
            **{
                "Weeks Rostered": ("Week", "nunique"),
                "Fantasy Points": ("Points", "sum"),
            }
        )
        .sort_values(["Season", "Weeks Rostered"], ascending=[False, False])
    )
    ownership["Fantasy Points"] = ownership["Fantasy Points"].round(2)
    summary = (
        ownership.groupby("Manager", as_index=False)
        .agg(
            Seasons=("Season", "nunique"),
            **{
                "Weeks Rostered": ("Weeks Rostered", "sum"),
                "Fantasy Points": ("Fantasy Points", "sum"),
            },
        )
        .sort_values("Weeks Rostered", ascending=False)
    )
    leader = summary.iloc[0]
    scope = f"in {season}" if season is not None else "in the loaded archive"
    if _contains(query, "how many owners", "how many managers"):
        owner_count = ownership["Manager"].nunique()
        answer = (
            f"{player_name} was rostered by {owner_count} "
            f"owner{'s' if owner_count != 1 else ''} {scope}."
        )
    else:
        answer = (
            f"{leader['Manager']} rostered {player_name} the longest {scope}: "
            f"{int(leader['Weeks Rostered'])} weeks."
        )
    return QueryResult(
        answer,
        title=f"{player_name} ownership history",
        table=ownership,
        interpretation=f"Fantasy owners who rostered {player_name}",
        suggestions=(
            f"How many times has {player_name} been traded?",
            f"Show me {player_name}'s full history.",
        ),
    )


def _answer_trade_query(
    query: str,
    season: int | None,
    player_history: pd.DataFrame,
    team_seasons: pd.DataFrame,
) -> QueryResult | None:
    if not _is_trade_query(query):
        return None
    if player_history.empty:
        return _player_required(
            "That trade question needs weekly player roster data."
        )

    trades = inferred_trade_analysis(player_history, team_seasons)
    trades = _season_filter(trades, season)
    player_name = _matching_player(query, player_history)
    managers = _matching_managers(query, team_seasons["Manager"])

    if player_name is not None:
        selected = trades[
            trades.apply(
                lambda row: _player_in_trade(row, player_name),
                axis=1,
            )
        ]
        season_text = (
            f" in {season}" if season is not None else " in the loaded archive"
        )
        count = len(selected)
        if count == 0:
            return QueryResult(
                f"{player_name} appears in 0 inferred trades{season_text}. "
                "This uses reciprocal week-to-week roster moves because ESPN "
                "does not provide a complete historical trade ledger.",
                title=f"{player_name} trade history",
                table=selected,
                interpretation=f"Inferred trade count for {player_name}",
                suggestions=(
                    f"Who rostered {player_name} the longest?",
                    f"Show me {player_name}'s full history.",
                ),
            )
        if _contains(
            query,
            "who traded",
            "traded to",
            "traded from",
            "what was",
            "what did",
            "in exchange",
            "return package",
        ):
            details = [
                _describe_player_trade(row, player_name)
                for _, row in selected.iterrows()
            ]
            return QueryResult(
                " ".join(details),
                title=f"{player_name} trade details",
                table=selected,
                interpretation=f"Inferred trade details for {player_name}",
                suggestions=(
                    f"Who rostered {player_name} the longest?",
                    f"Show me {player_name}'s full history.",
                ),
            )
        noun = "trade" if count == 1 else "trades"
        return QueryResult(
            f"{player_name} appears in {count} inferred {noun}{season_text}. "
            "The result is reconstructed from reciprocal roster moves.",
            title=f"{player_name} trade history",
            table=selected,
            interpretation=f"Inferred trade count for {player_name}",
            suggestions=(
                f"Who rostered {player_name} the longest?",
                f"Show me {player_name}'s full history.",
            ),
        )

    if trades.empty:
        return QueryResult(
            "No reciprocal roster moves could be identified as inferred trades "
            "in the loaded seasons.",
            title="Trade history",
        )

    if managers:
        manager = managers[0]
        selected = trades[
            (trades["Manager A"] == manager)
            | (trades["Manager B"] == manager)
        ]
        if selected.empty:
            return QueryResult(
                f"{manager} appears in 0 inferred trades in the matching seasons.",
                title=f"{manager} trade history",
                table=selected,
                interpretation=f"Post-trade value record for {manager}",
            )
        wins = int((selected["Winner"] == manager).sum())
        ties = int((selected["Winner"] == "Tie").sum())
        losses = len(selected) - wins - ties
        return QueryResult(
            f"{manager} made {len(selected)} inferred trades and went "
            f"{wins}-{losses}-{ties} in post-trade value.",
            title=f"{manager} trade history",
            table=selected,
            interpretation=f"Post-trade value record for {manager}",
        )

    if _contains(
        query,
        "biggest",
        "largest",
        "best trade",
        "most lopsided",
        "won the most",
        "best trader",
    ):
        if _contains(query, "won the most", "best trader"):
            winners = trades[trades["Winner"] != "Tie"]
            leaders = (
                winners.groupby("Winner", as_index=False)
                .size()
                .rename(columns={"size": "Trade Wins"})
                .sort_values("Trade Wins", ascending=False)
            )
            row = leaders.iloc[0]
            return QueryResult(
                f"{row['Winner']} has won the most inferred trades with "
                f"{int(row['Trade Wins'])}.",
                title="Post-trade value leaders",
                table=leaders,
            )
        row = trades.loc[trades["Value Margin"].idxmax()]
        return QueryResult(
            f"{row['Winner']} won the most lopsided inferred trade by "
            f"{row['Value Margin']:.2f} post-trade fantasy points in Week "
            f"{int(row['Trade Week'])}, {int(row['Season'])}.",
            title="Biggest post-trade value win",
            table=pd.DataFrame([row]),
        )

    return QueryResult(
        f"I found {len(trades)} inferred trades in the matching seasons. "
        "These are reconstructed from reciprocal week-to-week roster moves.",
        title="Trade history",
        table=trades,
    )


def _answer_draft_query(
    query: str,
    season: int | None,
    drafts: pd.DataFrame,
    player_history: pd.DataFrame,
    team_seasons: pd.DataFrame,
) -> QueryResult | None:
    draft_terms = (
        "draft",
        "drafted",
        "auction",
        "paid",
        "spent",
        "expensive",
        "value pick",
        "points per dollar",
        "points/$",
    )
    if not _contains(query, *draft_terms):
        return None
    if drafts.empty:
        return QueryResult("No draft history has been imported.")

    available = set(drafts["Season"].astype(int))
    if season is None:
        explicit = {
            int(year) for year in re.findall(r"\b(20\d{2})\b", query)
        } & available
        season = (
            max(explicit)
            if explicit
            else _relative_draft_season(query, available)
        )
    selected = _season_filter(drafts, season)
    position = _extract_position(query)
    if position is not None:
        selected = selected[selected["Position"] == position]
    managers = _matching_managers(query, team_seasons["Manager"])
    if managers:
        selected = selected[selected["Owner"] == managers[0]]
    player_name = _matching_draft_player(query, drafts)
    if player_name is not None:
        selected = selected[
            selected["Player"].map(_normalize_player_name)
            == _normalize_player_name(player_name)
        ]
    if selected.empty:
        return QueryResult("No matching draft purchases were found.")

    top_n = _extract_top_n(query)
    lowest = _contains(query, "cheapest", "least expensive", "lowest price")

    if _contains(
        query,
        "value",
        "points per dollar",
        "points/$",
        "bust",
        "return",
    ):
        if player_history.empty:
            return _player_required(
                "That draft-value question needs player box-score data."
            )
        values = all_draft_value(selected, player_history)
        values = values[values["Total Points"].notna()]
        if values.empty:
            return QueryResult(
                "No drafted players could be matched to the loaded player archive."
            )
        bust = _contains(query, "worst", "bust", "least value")
        values = values.sort_values(
            ["Points / $", "Total Points"],
            ascending=[bust, bust],
        ).head(top_n)
        row = values.iloc[0]
        direction = "worst" if bust else "best"
        return QueryResult(
            answer=(
                f"{row['Player']} was the {direction} draft value: "
                f"{row['Owner']} paid ${int(row['Price'])} in "
                f"{int(row['Season'])}, and the player scored "
                f"{row['Total Points']:.2f} points "
                f"({row['Points / $']:.2f} per dollar)."
            ),
            title=f"{direction.title()} draft values",
            table=values,
        )

    if player_name is not None or _contains(query, "who drafted"):
        purchases = selected.sort_values("Season")
        row = purchases.iloc[-1]
        if len(purchases) == 1:
            answer = (
                f"{row['Owner']} drafted {row['Player']} for "
                f"${int(row['Price'])} in {int(row['Season'])}."
            )
        else:
            answer = (
                f"{row['Player']} was drafted {len(purchases)} times in the "
                "matching seasons; the table shows each owner and price."
            )
        return QueryResult(
            answer=answer,
            title=f"{row['Player']} draft history",
            table=purchases,
        )

    if _contains(query, "spent", "spending"):
        grouped = selected.groupby("Owner", as_index=False).agg(
            Players=("Player", "count"),
            Spend=("Price", "sum"),
            **{"Average Price": ("Price", "mean")},
        )
        grouped["Average Price"] = grouped["Average Price"].round(2)
        grouped = grouped.sort_values("Spend", ascending=lowest).head(top_n)
        row = grouped.iloc[0]
        position_text = f" on {position}s" if position else ""
        direction = "least" if lowest else "most"
        return QueryResult(
            answer=(
                f"{row['Owner']} spent the {direction}{position_text}: "
                f"${int(row['Spend'])} across {int(row['Players'])} players."
            ),
            title="Draft spending",
            table=grouped,
        )

    purchases = selected.sort_values(
        ["Price", "Season"], ascending=[lowest, False]
    ).head(top_n)
    row = purchases.iloc[0]
    direction = "cheapest" if lowest else "most expensive"
    return QueryResult(
        answer=(
            f"{row['Player']} was the {direction} matching draft purchase: "
            f"{row['Owner']} paid ${int(row['Price'])} in "
            f"{int(row['Season'])}."
        ),
        title=f"{direction.title()} draft purchases",
        table=purchases,
    )


def _answer_player_query(
    query: str,
    players: pd.DataFrame,
    player_history_empty: bool,
) -> QueryResult | None:
    position = _extract_position(query)
    player_name = _matching_player(query, players)
    stat = _extract_stat(query)
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
        or stat is not None
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

    if stat is not None:
        column, label = stat
        if column not in filtered:
            return QueryResult(
                f"The loaded player archive does not contain {label}. Reload "
                "the archive to fetch the expanded player statistics."
            )
        if weekly:
            board = filtered.sort_values(column, ascending=lowest).head(top_n)
            row = board.iloc[0]
            return QueryResult(
                answer=(
                    f"{row['Player']} had the most {label} in a week with "
                    f"{_format_stat(row[column])} in Week {int(row['Week'])}, "
                    f"{int(row['Season'])}."
                ),
                title=f"Weekly {label} leaders",
                table=board,
            )
        totals = (
            filtered.groupby(
                ["Season", "Player", "Position"],
                as_index=False,
                dropna=False,
            )
            .agg(
                **{
                    "NFL Team": ("NFL Team", _latest_nonempty),
                    "Managers": ("Manager", _unique),
                    column: (column, "sum"),
                    "Weeks Rostered": ("Week", "nunique"),
                }
            )
            .sort_values(column, ascending=lowest)
            .head(top_n)
        )
        row = totals.iloc[0]
        if player_name is not None:
            answer = (
                f"{row['Player']} recorded {_format_stat(row[column])} "
                f"{label} in {int(row['Season'])}."
            )
        else:
            tied = totals[totals[column] == row[column]]["Player"].tolist()
            leaders = " and ".join(tied[:3])
            verb = "tied for the most" if len(tied) > 1 else "had the most"
            answer = (
                f"{leaders} {verb} {label} in {int(row['Season'])} with "
                f"{_format_stat(row[column])} each."
                if len(tied) > 1
                else f"{leaders} {verb} {label} in {int(row['Season'])} with "
                f"{_format_stat(row[column])}."
            )
        return QueryResult(
            answer=answer,
            title=f"Season {label} leaders",
            table=totals,
        )

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
        "performances, scoring leaders, lineup decisions, draft prices, "
        "position spending, or post-draft value.",
        title="Try another question",
    )


def _player_required(
    message: str = "That question needs player box-score data.",
) -> QueryResult:
    return QueryResult(
        f"{message} Open the Query Tool or another player-data section and "
        "reload the archive.",
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
    if matches:
        return max(matches)
    if _contains(query, "last year", "last season", "latest season"):
        return max(available) if available else None
    return None


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
    matches = _matching_players(query, players)
    return matches[0] if matches else None


def _matching_players(query: str, players: pd.DataFrame) -> list[str]:
    if players.empty or "Player" not in players:
        return []
    normalized_query = _normalize_player_name(query)
    names = sorted(
        set(players["Player"].dropna().astype(str)),
        key=len,
        reverse=True,
    )
    matches = [
        name
        for name in names
        if _normalize_player_name(name) in normalized_query
    ]
    unique: list[str] = []
    seen: set[str] = set()
    for name in matches:
        normalized = _normalize_player_name(name)
        if normalized not in seen:
            unique.append(name)
            seen.add(normalized)
    return unique


def _matching_player_across_sources(
    query: str,
    players: pd.DataFrame,
    drafts: pd.DataFrame,
) -> str | None:
    player_name = _matching_player(query, players)
    return player_name or _matching_draft_player(query, drafts)


def _player_in_trade(row: pd.Series, player_name: str) -> bool:
    target = _normalize_player_name(player_name)
    for column in ("Manager A Received", "Manager B Received"):
        if _trade_side_contains(row.get(column, ""), target):
            return True
    return False


def _trade_side_contains(received: object, normalized_player: str) -> bool:
    return any(
        _normalize_player_name(name) == normalized_player
        for name in str(received).split(",")
    )


def _describe_player_trade(row: pd.Series, player_name: str) -> str:
    target = _normalize_player_name(player_name)
    manager_a_received = str(row["Manager A Received"])
    if _trade_side_contains(manager_a_received, target):
        receiver = row["Manager A"]
        sender = row["Manager B"]
        received = manager_a_received
        return_package = row["Manager B Received"]
    else:
        receiver = row["Manager B"]
        sender = row["Manager A"]
        received = row["Manager B Received"]
        return_package = row["Manager A Received"]
    return (
        f"In Week {int(row['Trade Week'])}, {int(row['Season'])}, {receiver} "
        f"acquired {received} from {sender}; {sender} received {return_package}."
    )


def _is_trade_query(query: str) -> bool:
    return _contains(
        query,
        "trade",
        "traded",
        "trades",
        "trading",
        "deal",
        "dealt",
        "swap",
        "swapped",
        "exchange",
        "exchanged",
    )


def _matching_draft_player(query: str, drafts: pd.DataFrame) -> str | None:
    if drafts.empty or "Player" not in drafts:
        return None
    normalized_query = _normalize_player_name(query)
    names = sorted(
        set(drafts["Player"].dropna().astype(str)),
        key=len,
        reverse=True,
    )
    return next(
        (
            name
            for name in names
            if _normalize_player_name(name) in normalized_query
        ),
        None,
    )


def _normalize_player_name(value: str) -> str:
    normalized = re.sub(
        r"\b(jr|sr|ii|iii|iv)\b\.?",
        "",
        value.casefold(),
    )
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _extract_top_n(query: str) -> int:
    match = re.search(r"\btop\s+(\d{1,2})\b", query)
    if not match:
        return 10
    return min(50, max(1, int(match.group(1))))


def _relative_draft_season(
    query: str,
    available: set[int],
) -> int | None:
    if not available:
        return None
    if _contains(query, "last year", "last season", "latest draft"):
        return max(available)
    return None


def _extract_stat(query: str) -> tuple[str, str] | None:
    for alias in sorted(STAT_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", query):
            return STAT_ALIASES[alias]
    return None


def _latest_nonempty(values: pd.Series) -> str:
    for value in reversed(values.dropna().astype(str).tolist()):
        if value:
            return value
    return ""


def _unique(values: pd.Series) -> str:
    return ", ".join(dict.fromkeys(values.dropna().astype(str)))


def _format_stat(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.2f}"
