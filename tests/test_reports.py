import pandas as pd

from reports import archive_html_report, csv_download


def test_csv_download_hides_owner_key():
    data = csv_download(
        pd.DataFrame(
            [
                {
                    "Owner Key": "secret-key",
                    "Manager": "Alex",
                    "Team": "Changing Team Name",
                    "Fantasy Team": "Another Team Name",
                    "Wins": 10,
                }
            ]
        )
    ).decode()

    assert "Owner Key" not in data
    assert "secret-key" not in data
    assert "Changing Team Name" not in data
    assert "Another Team Name" not in data
    assert "Owner,Wins" in data


def test_html_report_builds_without_player_data():
    team_seasons = pd.DataFrame(
        [
            {
                "Season": 2025,
                "Owner Key": "alex",
                "Manager": "Alex",
                "Team": "A",
                "Wins": 10,
                "Losses": 4,
                "Ties": 0,
                "Champion": True,
                "Playoff Appearance": True,
                "Points For": 1500.0,
                "Points Against": 1400.0,
                "Final Standing": 1,
                "Acquisitions": 20,
                "Drops": 20,
                "Trades": 2,
                "FAAB Spent": 100,
            }
        ]
    )
    team_history = pd.DataFrame(
        [
            {
                "Season": 2025,
                "Week": 1,
                "Owner Key": "alex",
                "Manager": "Alex",
                "Team": "A",
                "Score": 120.0,
                "Opponent Owner Key": "unknown",
                "Opponent Manager": "Unknown",
                "Opponent": "Bye",
                "Opponent Score": None,
                "Margin": None,
                "Result": "W",
                "Season Phase": "Regular season",
            }
        ]
    )

    report = archive_html_report(
        "2025", team_history, team_seasons, pd.DataFrame()
    )

    assert "Phantasy Phootball History" in report
    assert "All-Time Owners" in report
    assert "secret-key" not in report
    assert ">A<" not in report
    assert ">Bye<" not in report
