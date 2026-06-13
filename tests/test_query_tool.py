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
