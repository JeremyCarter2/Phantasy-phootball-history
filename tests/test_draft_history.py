import pandas as pd

from draft_history import (
    draft_value,
    load_all_drafts,
    load_draft_history,
    owner_draft_summary,
    position_spend,
)


def test_2025_draft_parses_all_purchases_and_budgets():
    draft = load_draft_history(2025)
    summary = owner_draft_summary(draft)

    assert len(draft) == 180
    assert set(summary["Players"]) == {15}
    assert set(summary["Spend"]) == {200}
    assert "Kyle Hockert" in set(draft["Owner"])
    assert "Matthew Sheets" in set(draft["Owner"])


def test_all_historical_draft_exports_parse():
    expected_rows = {
        2020: 180,
        2021: 190,
        2022: 192,
        2023: 191,
        2024: 192,
        2025: 180,
    }
    for season, rows in expected_rows.items():
        draft = load_draft_history(season)
        assert len(draft) == rows
        assert draft["Owner"].nunique() == 12
        assert set(draft["Position"]) == {"QB", "RB", "WR", "TE", "D/ST"}

    combined = load_all_drafts()
    assert len(combined) == sum(expected_rows.values())
    assert set(combined["Season"]) == set(expected_rows)


def test_2025_draft_player_fields_and_position_spend():
    draft = load_draft_history(2025)
    fields = draft[draft["Player"] == "Bijan Robinson"].iloc[0]
    spend = position_spend(draft)

    assert fields["Owner"] == "Kevan Acker"
    assert fields["Position"] == "RB"
    assert fields["NFL Team"] == "ATL"
    assert fields["Bye Week"] == 5
    assert fields["Price"] == 48
    assert not spend.empty


def test_draft_value_is_empty_without_player_history():
    draft = load_draft_history(2025)

    assert draft_value(draft, draft.iloc[0:0]).empty


def test_draft_value_matches_suffixes_and_defense_names():
    draft = pd.DataFrame(
        [
            {
                "Season": 2025, "Owner": "Alex Jeli", "Purchase #": 1,
                "Player": "Patrick Mahomes II", "Position": "QB",
                "NFL Team": "KC", "Bye Week": 10, "Price": 20,
            },
            {
                "Season": 2025, "Owner": "Alex Jeli", "Purchase #": 2,
                "Player": "Dallas Cowboys", "Position": "D/ST",
                "NFL Team": "DST", "Bye Week": 10, "Price": 1,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "Season": 2025, "Week": 1, "Player": "Patrick Mahomes",
                "Position": "QB", "NFL Team": "KC", "Manager": "Alex Jeli",
                "Fantasy Team": "A", "Points": 20.0,
            },
            {
                "Season": 2025, "Week": 1, "Player": "Cowboys D/ST",
                "Position": "D/ST", "NFL Team": "DAL", "Manager": "Alex Jeli",
                "Fantasy Team": "A", "Points": 10.0,
            },
        ]
    )

    value = draft_value(draft, history)

    assert value["Total Points"].notna().all()
