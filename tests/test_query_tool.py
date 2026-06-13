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
