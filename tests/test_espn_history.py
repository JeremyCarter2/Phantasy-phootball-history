from types import SimpleNamespace

from espn_history import (
    PlayerSeasonResult,
    SeasonResult,
    build_dataframe,
    build_player_dataframe,
    build_scoring_leaders_dataframe,
    extract_performances,
    extract_player_performances,
)


def make_teams():
    alpha = SimpleNamespace(
        team_name="Alpha",
        scores=[101.25, 145.5],
        outcomes=["L", "W"],
        schedule=[],
    )
    beta = SimpleNamespace(
        team_name="Beta",
        scores=[110.0, 99.25],
        outcomes=["W", "L"],
        schedule=[],
    )
    alpha.schedule = [beta, beta]
    beta.schedule = [alpha, alpha]
    return [alpha, beta]


def test_extract_performances_includes_opponent_and_phase():
    rows = extract_performances(
        teams=make_teams(),
        season=2025,
        regular_season_weeks=1,
    )

    alpha_week_two = next(
        row for row in rows if row["Team"] == "Alpha" and row["Week"] == 2
    )
    assert alpha_week_two["Score"] == 145.5
    assert alpha_week_two["Opponent"] == "Beta"
    assert alpha_week_two["Opponent Score"] == 99.25
    assert alpha_week_two["Margin"] == 46.25
    assert alpha_week_two["Season Phase"] == "Playoffs"


def test_build_dataframe_orders_highest_score_first():
    result = SeasonResult(
        season=2025,
        league_name="Test League",
        performances=extract_performances(make_teams(), 2025, 1),
        team_seasons=[],
    )

    frame = build_dataframe([result])

    assert frame.iloc[0]["Team"] == "Alpha"
    assert frame.iloc[0]["Score"] == 145.5
    assert len(frame) == 4


def test_extract_player_performances_marks_starters_and_bench():
    team = SimpleNamespace(team_name="Alpha")
    starter = SimpleNamespace(
        playerId=1,
        name="Starter One",
        position="QB",
        proTeam="NE",
        points=31.4,
        slot_position="QB",
    )
    bench = SimpleNamespace(
        playerId=2,
        name="Bench Two",
        position="RB",
        proTeam="CHI",
        points=22.1,
        slot_position="BE",
    )
    box_score = SimpleNamespace(
        home_team=team,
        home_lineup=[starter, bench],
        away_team=None,
        away_lineup=[],
    )

    rows = extract_player_performances([box_score], 2025, 2, 14)

    assert rows[0]["Player"] == "Starter One"
    assert rows[0]["Lineup Status"] == "Starter"
    assert rows[1]["Lineup Status"] == "Bench/IR"


def test_build_player_dataframe_orders_highest_points_first():
    result = PlayerSeasonResult(
        season=2025,
        league_name="Test League",
        performances=[
                    {"Season": 2025, "Week": 1, "Player": "Low", "Position": "WR",
                     "Player ID": 1,
                     "NFL Team": "NE", "Fantasy Team": "Alpha", "Points": 10.0,
                     "Lineup Slot": "WR", "Eligible Slots": "WR|RB/WR/TE",
                     "Lineup Status": "Starter",
                     "Season Phase": "Regular season"},
                    {"Season": 2025, "Week": 1, "Player": "High", "Position": "QB",
                     "Player ID": 2,
                     "NFL Team": "BUF", "Fantasy Team": "Beta", "Points": 40.0,
                     "Lineup Slot": "QB", "Eligible Slots": "QB|OP",
                     "Lineup Status": "Starter",
             "Season Phase": "Regular season"},
        ],
    )

    frame = build_player_dataframe([result])

    assert frame.iloc[0]["Player"] == "High"
    assert frame.iloc[0]["Points"] == 40.0


def test_build_scoring_leaders_sums_player_season():
    history = build_player_dataframe(
        [
            PlayerSeasonResult(
                season=2025,
                league_name="Test League",
                performances=[
                    {"Season": 2025, "Week": 1, "Player": "Player A",
                     "Player ID": 1,
                     "Position": "RB", "NFL Team": "DET",
                     "Fantasy Team": "Alpha", "Points": 20.0,
                     "Lineup Slot": "RB", "Eligible Slots": "RB|RB/WR/TE",
                     "Lineup Status": "Starter",
                     "Season Phase": "Regular season"},
                    {"Season": 2025, "Week": 2, "Player": "Player A",
                     "Player ID": 1,
                     "Position": "RB", "NFL Team": "DET",
                     "Fantasy Team": "Beta", "Points": 30.0,
                     "Lineup Slot": "BE", "Eligible Slots": "RB|RB/WR/TE",
                     "Lineup Status": "Bench/IR",
                     "Season Phase": "Regular season"},
                ],
            )
        ]
    )

    leaders = build_scoring_leaders_dataframe(history)

    assert leaders.iloc[0]["Total Points"] == 50.0
    assert leaders.iloc[0]["Average Points"] == 25.0
    assert leaders.iloc[0]["Best Week"] == 2
    assert leaders.iloc[0]["Fantasy Teams"] == "Beta, Alpha"
