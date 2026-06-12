import pandas as pd

from analytics import (
    all_play_history,
    inferred_trade_analysis,
    lineup_efficiency,
    luck_history,
    manager_history,
    rivalry_history,
)


def team_history():
    return pd.DataFrame(
        [
            {"Season": 2025, "Week": 1, "Owner Key": "alex",
             "Manager": "Alex", "Team": "A", "Score": 120.0,
             "Opponent Owner Key": "blair", "Opponent Manager": "Blair",
             "Opponent": "B", "Opponent Score": 100.0, "Margin": 20.0,
             "Result": "W", "Season Phase": "Regular season"},
            {"Season": 2025, "Week": 1, "Owner Key": "blair",
             "Manager": "Blair", "Team": "B", "Score": 100.0,
             "Opponent Owner Key": "alex", "Opponent Manager": "Alex",
             "Opponent": "A", "Opponent Score": 120.0, "Margin": -20.0,
             "Result": "L", "Season Phase": "Regular season"},
            {"Season": 2025, "Week": 1, "Owner Key": "casey",
             "Manager": "Casey", "Team": "C", "Score": 110.0,
             "Opponent Owner Key": "drew", "Opponent Manager": "Drew",
             "Opponent": "D", "Opponent Score": 90.0, "Margin": 20.0,
             "Result": "W", "Season Phase": "Regular season"},
            {"Season": 2025, "Week": 1, "Owner Key": "drew",
             "Manager": "Drew", "Team": "D", "Score": 90.0,
             "Opponent Owner Key": "casey", "Opponent Manager": "Casey",
             "Opponent": "C", "Opponent Score": 110.0, "Margin": -20.0,
             "Result": "L", "Season Phase": "Regular season"},
        ]
    )


def team_seasons():
    return pd.DataFrame(
        [
            {"Season": 2025, "Owner Key": "alex", "Team": "A",
             "Manager": "Alex", "Wins": 1,
             "Losses": 0, "Ties": 0, "Champion": True,
             "Playoff Appearance": True, "Points For": 120.0,
             "Points Against": 100.0, "Final Standing": 1},
            {"Season": 2025, "Owner Key": "blair", "Team": "B",
             "Manager": "Blair", "Wins": 0,
             "Losses": 1, "Ties": 0, "Champion": False,
             "Playoff Appearance": False, "Points For": 100.0,
             "Points Against": 120.0, "Final Standing": 4},
            {"Season": 2025, "Owner Key": "casey", "Team": "C",
             "Manager": "Casey", "Wins": 1,
             "Losses": 0, "Ties": 0, "Champion": False,
             "Playoff Appearance": True, "Points For": 110.0,
             "Points Against": 90.0, "Final Standing": 2},
            {"Season": 2025, "Owner Key": "drew", "Team": "D",
             "Manager": "Drew", "Wins": 0,
             "Losses": 1, "Ties": 0, "Champion": False,
             "Playoff Appearance": False, "Points For": 90.0,
             "Points Against": 110.0, "Final Standing": 3},
        ]
    )


def test_manager_history_and_all_play():
    managers = manager_history(team_seasons())
    all_play = all_play_history(team_history())

    assert managers.iloc[0]["Manager"] == "Alex"
    assert all_play.loc[all_play["Manager"] == "Alex", "All-Play Wins"].iloc[0] == 3


def test_luck_and_rivalries():
    luck = luck_history(team_history(), team_seasons())
    rivalries = rivalry_history(team_history())

    assert "Expected Wins" in luck
    assert len(rivalries) == 2


def test_lineup_efficiency_uses_eligible_bench_player():
    players = pd.DataFrame(
        [
            {"Season": 2025, "Week": 1, "Owner Key": "alex",
             "Manager": "Alex", "Fantasy Team": "A",
             "Points": 10.0, "Lineup Status": "Starter",
             "Lineup Slot": "RB", "Position": "RB",
             "Eligible Slots": "RB|RB/WR/TE"},
            {"Season": 2025, "Week": 1, "Owner Key": "alex",
             "Manager": "Alex", "Fantasy Team": "A",
             "Points": 20.0, "Lineup Status": "Bench/IR",
             "Lineup Slot": "BE", "Position": "RB",
             "Eligible Slots": "RB|RB/WR/TE"},
        ]
    )

    result = lineup_efficiency(players).iloc[0]

    assert result["Actual Points"] == 10.0
    assert result["Optimal Points"] == 20.0
    assert result["Points Left"] == 10.0


def test_inferred_trade_analysis_scores_received_players():
    history = pd.DataFrame(
        [
            {"Season": 2025, "Week": 1, "Player ID": 1, "Player": "One",
             "Owner Key": "alex", "Manager": "Alex",
             "Fantasy Team": "A", "Points": 10.0},
            {"Season": 2025, "Week": 1, "Player ID": 2, "Player": "Two",
             "Owner Key": "blair", "Manager": "Blair",
             "Fantasy Team": "B", "Points": 5.0},
            {"Season": 2025, "Week": 2, "Player ID": 1, "Player": "One",
             "Owner Key": "blair", "Manager": "Blair",
             "Fantasy Team": "B", "Points": 20.0},
            {"Season": 2025, "Week": 2, "Player ID": 2, "Player": "Two",
             "Owner Key": "alex", "Manager": "Alex",
             "Fantasy Team": "A", "Points": 30.0},
            {"Season": 2025, "Week": 3, "Player ID": 1, "Player": "One",
             "Owner Key": "blair", "Manager": "Blair",
             "Fantasy Team": "B", "Points": 15.0},
            {"Season": 2025, "Week": 3, "Player ID": 2, "Player": "Two",
             "Owner Key": "alex", "Manager": "Alex",
             "Fantasy Team": "A", "Points": 25.0},
        ]
    )

    trades = inferred_trade_analysis(history)

    assert len(trades) == 1
    assert trades.iloc[0]["Winner"] == "Alex"
    assert trades.iloc[0]["Manager A Value"] == 55.0
    assert trades.iloc[0]["Manager B Value"] == 35.0
