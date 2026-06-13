import pandas as pd

from draft_history import load_all_drafts
from query_tool import answer_query
from test_analytics import team_history, team_seasons


def test_highest_score_query_with_season():
    result = answer_query(
        "Who had the highest score in 2025?",
        team_history(),
        team_seasons(),
        pd.DataFrame(),
    )

    assert "Alex" in result.answer
    assert "120.00" in result.answer


def test_champion_and_owner_summary_queries():
    champion = answer_query(
        "Who was champion in 2025?",
        team_history(),
        team_seasons(),
        pd.DataFrame(),
    )
    owner = answer_query(
        "How did Blair do in 2025?",
        team_history(),
        team_seasons(),
        pd.DataFrame(),
    )

    assert "Alex" in champion.answer
    assert "0-1" in owner.answer


def test_player_query_requests_player_archive():
    result = answer_query(
        "Who was the highest scoring player in 2025?",
        team_history(),
        team_seasons(),
        pd.DataFrame(),
    )

    assert result.needs_players


def player_history():
    return pd.DataFrame(
        [
            {
                "Season": 2020, "Week": 1, "Player": "Alpha Receiver",
                "Position": "WR", "NFL Team": "MIN", "Manager": "Alex",
                "Fantasy Team": "A", "Points": 20.0,
                "Receptions": 7.0,
                "Lineup Status": "Starter",
            },
            {
                "Season": 2020, "Week": 2, "Player": "Alpha Receiver",
                "Position": "WR", "NFL Team": "MIN", "Manager": "Alex",
                "Fantasy Team": "A", "Points": 25.0,
                "Receptions": 8.0,
                "Lineup Status": "Starter",
            },
            {
                "Season": 2020, "Week": 1, "Player": "Beta Receiver",
                "Position": "WR", "NFL Team": "DET", "Manager": "Blair",
                "Fantasy Team": "B", "Points": 30.0,
                "Receptions": 12.0,
                "Lineup Status": "Starter",
            },
            {
                "Season": 2020, "Week": 1, "Player": "Gamma Runner",
                "Position": "RB", "NFL Team": "GB", "Manager": "Casey",
                "Fantasy Team": "C", "Points": 40.0,
                "Receptions": 2.0,
                "Lineup Status": "Starter",
            },
        ]
    )


def player_team_seasons():
    seasons = team_seasons().copy()
    seasons["Season"] = 2020
    return seasons


def trade_player_history():
    rows = []
    players = [
        (1, "Patrick Mahomes II", "alex", "Alex", "A"),
        (2, "Jared Goff", "blair", "Blair", "B"),
    ]
    for player_id, player, owner, manager, team in players:
        rows.append(
            {
                "Season": 2022,
                "Week": 4,
                "Player ID": player_id,
                "Player": player,
                "Position": "QB",
                "NFL Team": "KC" if player_id == 1 else "DET",
                "Owner Key": owner,
                "Manager": manager,
                "Fantasy Team": team,
                "Points": 20.0,
                "Lineup Status": "Starter",
            }
        )
        rows.append(
            {
                "Season": 2022,
                "Week": 5,
                "Player ID": player_id,
                "Player": player,
                "Position": "QB",
                "NFL Team": "KC" if player_id == 1 else "DET",
                "Owner Key": "blair" if owner == "alex" else "alex",
                "Manager": "Blair" if owner == "alex" else "Alex",
                "Fantasy Team": "B" if owner == "alex" else "A",
                "Points": 25.0 if player_id == 1 else 10.0,
                "Lineup Status": "Starter",
            }
        )
    return pd.DataFrame(rows)


def trade_team_seasons():
    seasons = team_seasons().copy()
    seasons["Season"] = 2022
    seasons["Trades"] = 1
    return seasons


def test_best_position_query_uses_season_totals():
    result = answer_query(
        "Who was the best WR in 2020?",
        team_history(),
        player_team_seasons(),
        player_history(),
    )

    assert "Alpha Receiver" in result.answer
    assert "45.00" in result.answer


def test_position_week_query_uses_single_game_score():
    result = answer_query(
        "What was the highest WR game in 2020?",
        team_history(),
        player_team_seasons(),
        player_history(),
    )

    assert "Beta Receiver" in result.answer
    assert "30.00" in result.answer


def test_named_player_query_returns_season_history():
    result = answer_query(
        "How many points did Alpha Receiver score in 2020?",
        team_history(),
        player_team_seasons(),
        player_history(),
    )

    assert "45.00" in result.answer


def test_named_player_trade_count_handles_suffix_variants():
    result = answer_query(
        "How many times has Patrick Mahomes been traded?",
        team_history(),
        trade_team_seasons(),
        trade_player_history(),
    )

    assert "1 inferred trade" in result.answer
    assert "Patrick Mahomes II" in result.title
    assert len(result.table) == 1


def test_named_player_with_no_inferred_trades_returns_zero():
    history = trade_player_history()
    extra = history[history["Player"] == "Patrick Mahomes II"].copy()
    extra["Player ID"] = 3
    extra["Player"] = "Travis Kelce"
    extra["Owner Key"] = "alex"
    extra["Manager"] = "Alex"
    extra["Fantasy Team"] = "A"
    result = answer_query(
        "How many times has Travis Kelce been traded?",
        team_history(),
        trade_team_seasons(),
        pd.concat([history, extra], ignore_index=True),
    )

    assert "0 inferred trades" in result.answer
    assert "historical trade ledger" in result.answer


def test_owner_trade_query_summarizes_post_trade_record():
    result = answer_query(
        "How has Blair done in trades?",
        team_history(),
        trade_team_seasons(),
        trade_player_history(),
    )

    assert "made 1 inferred trades" in result.answer
    assert "1-0-0" in result.answer


def test_trade_query_understands_swapped_synonym():
    result = answer_query(
        "How often was Patrick Mahomes swapped?",
        team_history(),
        trade_team_seasons(),
        trade_player_history(),
    )

    assert "1 inferred trade" in result.answer


def test_trade_detail_query_explains_both_sides():
    result = answer_query(
        "What was Patrick Mahomes traded for?",
        team_history(),
        trade_team_seasons(),
        trade_player_history(),
    )

    assert "Blair acquired Patrick Mahomes II" in result.answer
    assert "Alex received Jared Goff" in result.answer


def test_player_ownership_query_lists_managers():
    result = answer_query(
        "Who had Alpha Receiver in 2020?",
        team_history(),
        player_team_seasons(),
        player_history(),
    )

    assert "Alex" in result.answer
    assert "Weeks Rostered" in result.table
    assert result.interpretation


def test_player_ownership_query_counts_distinct_owners():
    result = answer_query(
        "How many owners had Alpha Receiver in 2020?",
        team_history(),
        player_team_seasons(),
        player_history(),
    )

    assert "1 owner" in result.answer


def test_player_comparison_query_compares_season_points():
    result = answer_query(
        "Compare Alpha Receiver vs Beta Receiver in 2020",
        team_history(),
        player_team_seasons(),
        player_history(),
    )

    assert "Alpha Receiver" in result.answer
    assert "45.00" in result.answer
    assert "versus" in result.interpretation


def test_player_timeline_merges_draft_and_trade_history():
    result = answer_query(
        "Show me Patrick Mahomes draft and trade history",
        team_history(),
        trade_team_seasons(),
        trade_player_history(),
        load_all_drafts(),
    )

    assert "drafted" in result.answer
    assert "1 inferred trade" in result.answer
    assert {"Drafted", "Inferred trade"} <= set(result.table["Event"])


def test_most_catches_last_year_uses_latest_loaded_season():
    result = answer_query(
        "What player had the most catches last year?",
        team_history(),
        player_team_seasons(),
        player_history(),
    )

    assert "Alpha Receiver" in result.answer
    assert "15" in result.answer


def test_stat_query_reports_tied_leaders():
    history = player_history()
    history.loc[history["Player"] == "Beta Receiver", "Receptions"] = 15.0
    result = answer_query(
        "Who had the most catches last year?",
        team_history(),
        player_team_seasons(),
        history,
    )

    assert "Alpha Receiver and Beta Receiver" in result.answer
    assert "tied" in result.answer


def test_draft_query_finds_most_expensive_purchase():
    result = answer_query(
        "What was the most expensive draft pick in 2023?",
        team_history(),
        player_team_seasons(),
        pd.DataFrame(),
        load_all_drafts(),
    )

    assert "Josh Allen" in result.answer
    assert "$59" in result.answer


def test_draft_query_summarizes_position_spending():
    result = answer_query(
        "Who spent the most on QBs in the 2025 draft?",
        team_history(),
        player_team_seasons(),
        pd.DataFrame(),
        load_all_drafts(),
    )

    assert "Owner" in result.table
    assert "Spend" in result.table


def test_draft_query_tracks_player_across_suffix_variants():
    result = answer_query(
        "Who drafted Patrick Mahomes each year?",
        team_history(),
        player_team_seasons(),
        pd.DataFrame(),
        load_all_drafts(),
    )

    assert len(result.table) >= 5
    assert set(result.table["Player"]) <= {
        "Patrick Mahomes",
        "Patrick Mahomes II",
    }


def test_draft_value_query_requires_player_data():
    result = answer_query(
        "What was the best value pick in the 2025 draft?",
        team_history(),
        player_team_seasons(),
        pd.DataFrame(),
        load_all_drafts(),
    )

    assert result.needs_players
