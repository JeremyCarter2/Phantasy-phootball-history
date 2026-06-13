from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from espn_history import build_scoring_leaders_dataframe


DRAFT_DIR = Path(__file__).parent / "data" / "drafts"
OWNER_NAMES = {
    "Alex": "Alex Jeli",
    "Austin": "Austin Briggs",
    "Cam": "Cameron Dixon",
    "Coleman": "Coleman Quinn",
    "Jaron": "Jaron Wilson",
    "Jeremy": "jeremy carter",
    "Kevan": "Kevan Acker",
    "Kyle": "Kyle Hockert",
    "Matt": "Matt Davis",
    "Ray": "Raymond McGuinness",
    "Sheets": "Matthew Sheets",
    "Timmy": "Timmy Stowell",
}
PLAYER_PATTERN = re.compile(
    r"^(?P<player>.+)\n(?P<position>[^-]+?)\s*-\s*"
    r"(?P<nfl_team>[^-]*?)\s*-\s*(?P<bye>\d*)$",
    re.DOTALL,
)


def available_draft_seasons() -> list[int]:
    return sorted(
        [
            int(path.stem)
            for path in DRAFT_DIR.glob("*.csv")
            if path.stem.isdigit()
        ],
        reverse=True,
    )


def load_all_drafts() -> pd.DataFrame:
    drafts = [load_draft_history(season) for season in available_draft_seasons()]
    drafts = [draft for draft in drafts if not draft.empty]
    return pd.concat(drafts, ignore_index=True) if drafts else _empty_draft()


def load_draft_history(season: int) -> pd.DataFrame:
    path = DRAFT_DIR / f"{season}.csv"
    if not path.exists():
        return _empty_draft()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return _empty_draft()

    header_index, owner_columns = _find_owner_columns(rows)
    if not owner_columns:
        return _empty_draft()
    purchases = []
    for owner_column, draft_owner in owner_columns:
        owner = _owner_name(draft_owner)
        purchase_number = 0
        for row in rows[header_index + 1 :]:
            if owner_column + 1 >= len(row):
                continue
            player_text = row[owner_column].strip()
            price_text = row[owner_column + 1].strip()
            if not PLAYER_PATTERN.match(player_text) or not price_text.startswith("$"):
                continue
            purchase_number += 1
            parsed = _parse_player(player_text)
            purchases.append(
                {
                    "Season": season,
                    "Owner": owner,
                    "Purchase #": purchase_number,
                    **parsed,
                    "Price": _parse_price(price_text),
                }
            )
    return pd.DataFrame(purchases, columns=_empty_draft().columns)


def owner_draft_summary(draft: pd.DataFrame) -> pd.DataFrame:
    if draft.empty:
        return pd.DataFrame()
    summary = draft.groupby("Owner", as_index=False).agg(
        Players=("Player", "count"),
        Spend=("Price", "sum"),
        **{
            "Average Price": ("Price", "mean"),
            "Highest Price": ("Price", "max"),
        },
    )
    summary["Average Price"] = summary["Average Price"].round(2)
    return summary.sort_values(
        ["Spend", "Highest Price"], ascending=False, ignore_index=True
    )


def position_spend(draft: pd.DataFrame) -> pd.DataFrame:
    if draft.empty:
        return pd.DataFrame()
    grouped = draft.groupby(["Owner", "Position"], as_index=False).agg(
        Players=("Player", "count"),
        Spend=("Price", "sum"),
    )
    return grouped.sort_values(
        ["Owner", "Spend"], ascending=[True, False], ignore_index=True
    )


def draft_value(
    draft: pd.DataFrame,
    player_history: pd.DataFrame,
) -> pd.DataFrame:
    if draft.empty or player_history.empty:
        return pd.DataFrame()
    season = int(draft["Season"].iloc[0])
    leaders = build_scoring_leaders_dataframe(
        player_history[player_history["Season"] == season]
    )
    draft_keys = draft.copy()
    leader_keys = leaders.copy()
    draft_keys["_Match Key"] = draft_keys.apply(_match_key, axis=1)
    leader_keys["_Match Key"] = leader_keys.apply(_match_key, axis=1)
    merged = draft_keys.merge(
        leader_keys[
            [
                "_Match Key",
                "Total Points",
                "Average Points",
                "Best Week Points",
                "Weeks Rostered",
            ]
        ],
        on="_Match Key",
        how="left",
    ).drop(columns="_Match Key")
    merged["Points / $"] = (
        merged["Total Points"] / merged["Price"]
    ).round(2)
    return merged.sort_values(
        ["Points / $", "Total Points"],
        ascending=False,
        ignore_index=True,
    )


def owner_draft_value(value: pd.DataFrame) -> pd.DataFrame:
    if value.empty:
        return pd.DataFrame()
    summary = value.groupby("Owner", as_index=False).agg(
        Spend=("Price", "sum"),
        **{
            "Drafted Points": ("Total Points", "sum"),
            "Matched Players": ("Total Points", "count"),
        },
    )
    summary["Points / $"] = (
        summary["Drafted Points"] / summary["Spend"]
    ).round(2)
    summary["Drafted Points"] = summary["Drafted Points"].round(2)
    return summary.sort_values(
        "Points / $", ascending=False, ignore_index=True
    )


def _parse_player(value: str) -> dict[str, object]:
    match = PLAYER_PATTERN.match(value.strip())
    if not match:
        return {
            "Player": value.replace("\n", " ").strip(),
            "Position": "",
            "NFL Team": "",
            "Bye Week": None,
        }
    bye = match.group("bye")
    position = match.group("position").strip()
    return {
        "Player": match.group("player").strip(),
        "Position": "D/ST" if position in {"DS", "DST"} else position,
        "NFL Team": match.group("nfl_team").strip(),
        "Bye Week": int(bye) if bye else None,
    }


def _parse_price(value: str) -> int:
    digits = re.sub(r"\D", "", value)
    return int(digits) if digits else 0


def _find_owner_columns(
    rows: list[list[str]],
) -> tuple[int, list[tuple[int, str]]]:
    best_index = -1
    best_columns: list[tuple[int, str]] = []
    for row_index, row in enumerate(rows[:6]):
        columns = [
            (column, value.strip())
            for column, value in enumerate(row)
            if _owner_name(value.strip()) in OWNER_NAMES.values()
        ]
        if len(columns) > len(best_columns):
            best_index = row_index
            best_columns = columns
    return best_index, best_columns


def _owner_name(value: str) -> str:
    normalized = value.strip().upper().rstrip(".")
    aliases = {
        **{key.upper(): owner for key, owner in OWNER_NAMES.items()},
        "MATT D": "Matt Davis",
    }
    return aliases.get(normalized, value.strip())


def _match_key(row: pd.Series) -> str:
    player = str(row["Player"])
    position = str(row["Position"])
    if position == "D/ST":
        words = re.findall(r"[a-z0-9]+", player.casefold())
        mascot = (
            words[-3]
            if len(words) >= 3 and words[-2:] == ["d", "st"]
            else words[-1]
        )
        return f"d/st:{mascot}"
    normalized = re.sub(
        r"\b(jr|sr|ii|iii|iv)\b\.?",
        "",
        player.casefold(),
    )
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return f"{position.casefold()}:{normalized}"


def _empty_draft() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Season",
            "Owner",
            "Purchase #",
            "Player",
            "Position",
            "NFL Team",
            "Bye Week",
            "Price",
        ]
    )
