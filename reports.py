from __future__ import annotations

from html import escape

import pandas as pd

from analytics import (
    inferred_trade_analysis,
    luck_history,
    manager_history,
    rivalry_history,
    transaction_summary,
)
from espn_history import build_scoring_leaders_dataframe


HIDDEN_COLUMNS = {
    "Owner Key",
    "Owner A Key",
    "Owner B Key",
    "Opponent Owner Key",
}


def csv_download(frame: pd.DataFrame) -> bytes:
    return _display_frame(frame).to_csv(index=False).encode("utf-8")


def archive_html_report(
    season_label: str,
    team_history: pd.DataFrame,
    team_seasons: pd.DataFrame,
    player_history: pd.DataFrame,
) -> str:
    sections = [
        ("All-Time Owners", manager_history(team_seasons)),
        ("Luck and All-Play", luck_history(team_history, team_seasons)),
        ("Rivalries", rivalry_history(team_history).head(100)),
        ("Transaction Activity", transaction_summary(team_seasons)),
        (
            "Highest Team Scores",
            team_history.sort_values("Score", ascending=False).head(50),
        ),
    ]
    if not player_history.empty:
        sections.extend(
            [
                (
                    "Player Season Leaders",
                    build_scoring_leaders_dataframe(player_history).head(100),
                ),
                (
                    "Trade Winners and Losers",
                    inferred_trade_analysis(
                        player_history, team_seasons
                    ).head(100),
                ),
            ]
        )

    body = "".join(
        f"<section><h2>{escape(title)}</h2>"
        f"{_display_frame(frame).to_html(index=False, border=0)}</section>"
        for title, frame in sections
        if not frame.empty
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phantasy Phootball Report {escape(season_label)}</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#17211b;background:#f7f5ef}}
h1{{font-size:36px;margin-bottom:4px}} h2{{margin-top:40px}}
p{{color:#5f665f}} table{{width:100%;border-collapse:collapse;background:white}}
th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left;font-size:13px}}
th{{background:#386b52;color:white;position:sticky;top:0}}
section{{overflow-x:auto}} 
</style>
</head>
<body>
<h1>Phantasy Phootball History</h1>
<p>League 682600 | Seasons {escape(season_label)}</p>
{body}
</body>
</html>"""


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[column for column in HIDDEN_COLUMNS if column in frame.columns]
    ).rename(
        columns={
            "Manager": "Owner",
            "Managers": "Owners",
            "Opponent Manager": "Opponent Owner",
            "Manager A": "Owner A",
            "Manager B": "Owner B",
            "Manager A Wins": "Owner A Wins",
            "Manager B Wins": "Owner B Wins",
            "Manager A Points": "Owner A Points",
            "Manager B Points": "Owner B Points",
            "Manager A Received": "Owner A Received",
            "Manager B Received": "Owner B Received",
            "Manager A Value": "Owner A Value",
            "Manager B Value": "Owner B Value",
        }
    )
