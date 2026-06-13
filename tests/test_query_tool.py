import pandas as pd

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
                "Lineup Status": "Starter",
            },
            {
                "Season": 2020, "Week": 2, "Player": "Alpha Receiver",
                "Position": "WR", "NFL Team": "MIN", "Manager": "Alex",
                "Fantasy Team": "A", "Points": 25.0,
                "Lineup Status": "Starter",
            },
            {
                "Season": 2020, "Week": 1, "Player": "Beta Receiver",
                "Position": "WR", "NFL Team": "DET", "Manager": "Blair",
                "Fantasy Team": "B", "Points": 30.0,
                "Lineup Status": "Starter",
            },
            {
                "Season": 2020, "Week": 1, "Player": "Gamma Runner",
                "Position": "RB", "NFL Team": "GB", "Manager": "Casey",
                "Fantasy Team": "C", "Points": 40.0,
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
