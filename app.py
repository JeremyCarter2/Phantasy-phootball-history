from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from analytics import (
    all_play_history,
    inferred_trade_analysis,
    league_records,
    lineup_efficiency,
    luck_history,
    manager_history,
    player_roster_history,
    rivalry_history,
    transaction_summary,
)
from espn_history import (
    LEAGUE_ID,
    PlayerSeasonResult,
    SeasonResult,
    build_dataframe,
    build_player_dataframe,
    build_scoring_leaders_dataframe,
    build_team_seasons_dataframe,
    fetch_player_season,
    fetch_season,
)
from reports import archive_html_report, csv_download
from query_tool import answer_query


PAGES = [
    "Overview",
    "Query Tool",
    "Leaderboards",
    "Owners",
    "Luck & All-Play",
    "Rivalries",
    "Records",
    "Lineup Decisions",
    "Transactions",
    "Player History",
    "Season Recap",
]
PLAYER_PAGES = {
    "Lineup Decisions",
    "Transactions",
    "Player History",
    "Season Recap",
}
DOWNLOAD_INDEX = 0
PLAYER_DATA_VERSION = 2

st.set_page_config(
    page_title="Phantasy Phootball History",
    page_icon=":football:",
    layout="wide",
)
st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at 85% 0%, rgba(65,118,91,.18), transparent 30rem),
          #f7f5ef;
      }
      h1, h2, h3 { letter-spacing: -.035em; }
      [data-testid="stMetric"] {
        background: rgba(255,255,255,.78);
        border: 1px solid rgba(31,41,55,.10);
        border-radius: 16px;
        padding: 1rem;
      }
      .hero { padding: 1.1rem 0 .5rem; }
      .hero h1 { color: #17211b !important; }
      .eyebrow {
        color: #386b52; font-size: .78rem; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase;
      }
      .subtle { color: #5f665f !important; max-width: 54rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def secret(name: str) -> str:
    try:
        return str(st.secrets[name]).strip()
    except (KeyError, FileNotFoundError):
        return ""


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_team_season(
    season: int,
    _espn_s2: str,
    _swid: str,
) -> SeasonResult:
    return fetch_season(season, _espn_s2, _swid)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_player_season(
    season: int,
    _espn_s2: str,
    _swid: str,
    _data_version: int = PLAYER_DATA_VERSION,
) -> PlayerSeasonResult:
    return fetch_player_season(season, _espn_s2, _swid)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_archive_cached(
    seasons: tuple[int, ...],
    _espn_s2: str,
    _swid: str,
    include_players: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    team_results: list[SeasonResult] = []
    player_results: list[PlayerSeasonResult] = []
    failures: list[dict[str, Any]] = []

    jobs: dict[Any, tuple[str, int]] = {}
    max_workers = min(4, max(1, len(seasons) * (2 if include_players else 1)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for season in seasons:
            jobs[
                executor.submit(load_team_season, season, _espn_s2, _swid)
            ] = ("team", season)
            if include_players:
                jobs[
                    executor.submit(
                        load_player_season,
                        season,
                        _espn_s2,
                        _swid,
                        PLAYER_DATA_VERSION,
                    )
                ] = ("player", season)

        for future in as_completed(jobs):
            kind, season = jobs[future]
            try:
                result = future.result()
                if kind == "team":
                    team_results.append(result)
                else:
                    player_results.append(result)
            except Exception as exc:
                failures.append(
                    {"season": season, "kind": kind, "error": str(exc)}
                )

    team_results.sort(key=lambda result: result.season)
    player_results.sort(key=lambda result: result.season)
    return (
        build_dataframe(team_results),
        build_team_seasons_dataframe(team_results),
        build_player_dataframe(player_results),
        failures,
    )


def load_archive(
    seasons: list[int],
    espn_s2: str,
    swid: str,
    include_players: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    with st.spinner(
        "Loading seasons in parallel. Cached seasons return almost instantly..."
    ):
        return load_archive_cached(
            tuple(seasons), espn_s2, swid, include_players
        )


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">League 682600 | The archive</div>
      <h1>Phantasy Phootball History</h1>
      <p class="subtle">
        Records, rivalries, luck, lineup decisions, player history, and enough
        evidence to keep league arguments alive indefinitely.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

espn_s2, swid = secret("ESPN_S2"), secret("SWID")
if not espn_s2 or not swid:
    st.error("The app owner must configure ESPN_S2 and SWID in Streamlit secrets.")
    st.stop()

current_year = date.today().year
with st.sidebar:
    st.header("League archive")
    page = st.radio("Explore", PAGES)
    start_year = st.number_input(
        "First season",
        min_value=2000,
        max_value=current_year,
        value=max(2019, current_year - 7),
    )
    end_year = st.number_input(
        "Last season",
        min_value=2000,
        max_value=current_year,
        value=current_year - 1,
    )
    query_players = (
        st.toggle("Include player and lineup questions", value=False)
        if page == "Query Tool"
        else False
    )
    include_players = page in PLAYER_PAGES or query_players or (
        page == "Leaderboards"
        and st.toggle("Include player leaderboards", value=False)
    )
    load_clicked = st.button("Load archive", type="primary", width="stretch")
    if st.button("Clear cached archive", width="stretch"):
        load_archive_cached.clear()
        load_team_season.clear()
        load_player_season.clear()
        st.session_state.pop("archive", None)
        st.rerun()
    if include_players:
        st.caption("Player and lineup views take longer to load.")
    st.caption("Unavailable ESPN seasons are skipped without stopping the archive.")

if start_year > end_year:
    st.warning("The first season must be before the last season.")
    st.stop()
if end_year - start_year + 1 > 20:
    st.warning("Choose 20 seasons or fewer.")
    st.stop()

if load_clicked:
    seasons = list(range(int(start_year), int(end_year) + 1))
    team_history, team_seasons, player_history, failures = load_archive(
        seasons, espn_s2, swid, include_players
    )
    st.session_state["archive"] = {
        "team_history": team_history,
        "team_seasons": team_seasons,
        "player_history": player_history,
        "failures": failures,
        "range": (int(start_year), int(end_year)),
        "has_players": include_players,
    }

archive = st.session_state.get("archive")
if archive is None:
    st.info("Choose a section and season range, then select **Load archive**.")
    st.stop()

team_history = archive["team_history"]
team_seasons = archive["team_seasons"]
player_history = archive["player_history"]
loaded_start, loaded_end = archive["range"]

if archive["failures"]:
    unavailable = ", ".join(str(item["season"]) for item in archive["failures"])
    st.warning(f"ESPN did not return usable data for: {unavailable}")
if team_history.empty:
    st.warning("No team history was available.")
    st.stop()
if page in PLAYER_PAGES and player_history.empty:
    st.info("This section needs player data. Select **Load archive** again.")
    st.stop()

season_label = (
    str(loaded_start)
    if loaded_start == loaded_end
    else f"{loaded_start}-{loaded_end}"
)


def show_table(frame: pd.DataFrame, height: int | None = None) -> None:
    global DOWNLOAD_INDEX
    DOWNLOAD_INDEX += 1
    display_frame = frame.drop(
        columns=[
            column
            for column in (
                "Owner Key",
                "Owner A Key",
                "Owner B Key",
                "Opponent Owner Key",
                "Team",
                "Opponent",
                "Fantasy Team",
                "Fantasy Teams",
                "Teams",
                "Team Names A",
                "Team Names B",
                "Team A",
                "Team B",
                "From Team",
                "To Team",
            )
            if column in frame.columns
        ]
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
    options: dict[str, Any] = {
        "hide_index": True,
        "width": "stretch",
        "column_config": {
            "Win %": st.column_config.NumberColumn(format="%.3f"),
            "All-Play %": st.column_config.NumberColumn(format="%.3f"),
            "Efficiency %": st.column_config.NumberColumn(format="%.1%%"),
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Points": st.column_config.NumberColumn(format="%.2f"),
            "Total Points": st.column_config.NumberColumn(format="%.2f"),
            "Average Points": st.column_config.NumberColumn(format="%.2f"),
            "Luck": st.column_config.NumberColumn(format="%+.2f"),
        },
    }
    if height is not None:
        options["height"] = height
    st.dataframe(display_frame, **options)
    st.download_button(
        "Download this table",
        data=csv_download(display_frame),
        file_name=f"phantasy-phootball-{page.lower().replace(' ', '-')}.csv",
        mime="text/csv",
        key=f"download-{page}-{DOWNLOAD_INDEX}",
    )


if page == "Overview":
    st.header(f"Archive Overview | {season_label}")
    owners = manager_history(team_seasons)
    records = league_records(team_history)
    champion_count = int(team_seasons["Champion"].sum())
    cols = st.columns(4)
    cols[0].metric("Seasons loaded", team_seasons["Season"].nunique())
    cols[1].metric("Owners", team_seasons["Owner Key"].nunique())
    cols[2].metric("Championships", champion_count)
    cols[3].metric("Matchup scores", len(team_history))

    st.subheader("All-time wins")
    wins_chart = owners.sort_values("Wins").set_index("Manager")["Wins"]
    st.bar_chart(wins_chart, horizontal=True, color="#386b52")

    st.subheader("Scoring by season")
    scoring_chart = (
        team_seasons.groupby("Season")["Points For"].mean().round(2)
    )
    st.line_chart(scoring_chart, color="#386b52")

    highest = records["Highest score"]
    st.info(
        f"Archive record: {highest['Manager']} scored {highest['Score']:.2f} "
        f"in Week {int(highest['Week'])}, {int(highest['Season'])}."
    )

elif page == "Query Tool":
    st.header(f"Ask the Archive | {season_label}")
    st.write(
        "Ask a plain-English question about the seasons you loaded. Answers "
        "are calculated directly from league data rather than generated by an AI."
    )
    st.caption(
        "Examples: “Who had the highest score in 2023?”, “Who has the most "
        "championships?”, “What is the rivalry between Alex Jeli and Cameron "
        "Dixon?”, “Who was the best WR in 2020?”, or “What was Justin "
        "Jefferson's highest-scoring week?”, or “Who had the most catches "
        "last year?”"
    )
    with st.form("archive-query"):
        question = st.text_input(
            "Question",
            placeholder="Who was the luckiest owner in 2024?",
        )
        submitted = st.form_submit_button("Search archive", type="primary")
    if submitted:
        st.session_state["last_query"] = question
    question = st.session_state.get("last_query", "")
    if question:
        result = answer_query(
            question,
            team_history,
            team_seasons,
            player_history,
        )
        st.subheader(result.title)
        if result.needs_players:
            st.warning(result.answer)
        else:
            st.success(result.answer)
        if not result.table.empty:
            show_table(result.table)

elif page == "Leaderboards":
    st.header(f"Leaderboards | {season_label}")
    choices = ["Weekly owner scores"]
    if not player_history.empty:
        choices += ["Weekly players", "Season scoring leaders"]
    leaderboard_type = st.radio("Leaderboard", choices, horizontal=True)
    top_n = st.slider("Show", 10, 100, 25, 5)

    if leaderboard_type == "Weekly owner scores":
        board = team_history.sort_values("Score", ascending=False).head(top_n)
        show_table(
            board[
                [
                    "Season", "Week", "Manager", "Score",
                    "Opponent Manager", "Opponent Score",
                    "Margin", "Result", "Season Phase",
                ]
            ]
        )
        chart = board.sort_values("Score").copy()
        chart["Label"] = (
            chart["Manager"]
            + " | "
            + chart["Season"].astype(str)
            + " W"
            + chart["Week"].astype(str)
        )
        st.bar_chart(
            chart.set_index("Label")["Score"],
            horizontal=True,
            color="#386b52",
        )
    elif leaderboard_type == "Weekly players":
        statuses = st.multiselect(
            "Lineup status", ["Starter", "Bench/IR"], default=["Starter"]
        )
        board = player_history[
            player_history["Lineup Status"].isin(statuses)
        ].sort_values("Points", ascending=False).head(top_n)
        show_table(
            board[
                [
                    "Season", "Week", "Player", "Position", "NFL Team",
                    "Manager", "Points", "Lineup Slot",
                    "Season Phase",
                ]
            ]
        )
        chart = board.sort_values("Points").copy()
        chart["Label"] = (
            chart["Player"]
            + " | "
            + chart["Season"].astype(str)
            + " W"
            + chart["Week"].astype(str)
        )
        st.bar_chart(
            chart.set_index("Label")["Points"],
            horizontal=True,
            color="#386b52",
        )
    else:
        board = build_scoring_leaders_dataframe(player_history)
        selected_season = st.selectbox(
            "Season", sorted(board["Season"].unique(), reverse=True)
        )
        show_table(
            board[board["Season"] == selected_season].head(top_n)
        )
        st.bar_chart(
            board[board["Season"] == selected_season]
            .head(top_n)
            .sort_values("Total Points")
            .set_index("Player")["Total Points"],
            horizontal=True,
            color="#386b52",
        )

elif page == "Owners":
    st.header(f"All-Time Owners | {season_label}")
    history = manager_history(team_seasons)
    leader = history.iloc[0]
    metrics = st.columns(3)
    metrics[0].metric("Most championships", leader["Manager"], leader["Championships"])
    metrics[1].metric("Most wins", history.loc[history["Wins"].idxmax(), "Manager"])
    metrics[2].metric(
        "Most points",
        history.loc[history["Points For"].idxmax(), "Manager"],
    )
    show_table(history)
    st.bar_chart(
        history.sort_values("Wins").set_index("Manager")["Wins"],
        horizontal=True,
        color="#386b52",
    )

elif page == "Luck & All-Play":
    st.header(f"Luck & All-Play | {season_label}")
    luck = luck_history(team_history, team_seasons)
    selected_season = st.selectbox(
        "Season", sorted(luck["Season"].unique(), reverse=True)
    )
    season_luck = luck[luck["Season"] == selected_season]
    show_table(
        season_luck[
            [
                "Manager", "Wins", "Expected Wins", "Luck",
                "All-Play Wins", "All-Play Losses", "All-Play %"
            ]
        ]
    )
    st.bar_chart(
        season_luck.sort_values("Luck").set_index("Manager")["Luck"],
        horizontal=True,
        color="#386b52",
    )
    st.caption("Luck = actual regular-season wins minus all-play expected wins.")

elif page == "Rivalries":
    st.header(f"Head-to-Head Rivalries | {season_label}")
    rivalries = rivalry_history(team_history)
    managers = sorted(
        set(rivalries["Manager A"]).union(rivalries["Manager B"])
    )
    selected = st.multiselect("Filter owners", managers)
    if selected:
        rivalries = rivalries[
            rivalries["Manager A"].isin(selected)
            | rivalries["Manager B"].isin(selected)
        ]
    show_table(rivalries)
    if not rivalries.empty:
        chart = rivalries.head(20).copy()
        chart["Rivalry"] = chart["Manager A"] + " vs " + chart["Manager B"]
        st.bar_chart(
            chart.set_index("Rivalry")["Games"],
            horizontal=True,
            color="#386b52",
        )

elif page == "Records":
    st.header(f"League Records | {season_label}")
    records = league_records(team_history)
    cols = st.columns(4)
    for column, (label, row) in zip(cols, records.items()):
        value = row["Score"] if label not in {"Largest win", "Closest game"} else abs(row["Margin"])
        column.metric(label, f"{value:.2f}", row["Manager"])
    show_table(
        team_history.sort_values("Score", ascending=False).head(50)[
            [
                "Season", "Week", "Manager", "Score",
                "Opponent Manager", "Margin", "Season Phase",
            ]
        ]
    )

elif page == "Lineup Decisions":
    st.header(f"Lineup Decisions | {season_label}")
    efficiency = lineup_efficiency(player_history)
    summary = efficiency.groupby(["Owner Key", "Manager"], as_index=False).agg(
        Weeks=("Week", "count"),
        **{
            "Actual Points": ("Actual Points", "sum"),
            "Optimal Points": ("Optimal Points", "sum"),
            "Points Left": ("Points Left", "sum"),
            "Bench Points": ("Bench Points", "sum"),
        },
    )
    summary["Efficiency %"] = summary["Actual Points"] / summary["Optimal Points"]
    show_table(summary.sort_values("Efficiency %", ascending=False))
    st.bar_chart(
        summary.sort_values("Efficiency %").set_index("Manager")["Efficiency %"],
        horizontal=True,
        color="#386b52",
    )
    st.subheader("Most painful weeks")
    show_table(efficiency.sort_values("Points Left", ascending=False).head(50))

elif page == "Transactions":
    st.header(f"Transaction Activity | {season_label}")
    show_table(transaction_summary(team_seasons))
    activity = transaction_summary(team_seasons)
    st.bar_chart(
        activity.sort_values("Acquisitions").set_index("Manager")[
            ["Acquisitions", "Trades"]
        ],
        horizontal=True,
    )
    st.subheader("Trade winners and losers")
    trades = inferred_trade_analysis(player_history, team_seasons)
    if trades.empty:
        st.info(
            "No reciprocal roster moves could be identified in the selected "
            "seasons. ESPN did not provide a usable official historical trade log."
        )
    else:
        selected_trade_season = st.selectbox(
            "Trade season", sorted(trades["Season"].unique(), reverse=True)
        )
        show_table(trades[trades["Season"] == selected_trade_season])
    st.caption(
        "Trade value is the fantasy points each received player scored while "
        "remaining on the receiving roster after the trade. Trades are inferred "
        "from reciprocal week-to-week roster moves because ESPN does not return "
        "a usable official historical trade log for this league."
    )

elif page == "Player History":
    st.header(f"Player Roster History | {season_label}")
    history = player_roster_history(player_history)
    query = st.text_input("Search player")
    if query:
        history = history[
            history["Player"].str.contains(query, case=False, na=False)
        ]
    show_table(history.head(250), height=650)

elif page == "Season Recap":
    st.header("Season Recap")
    season = st.selectbox(
        "Season", sorted(team_seasons["Season"].unique(), reverse=True)
    )
    season_teams = team_seasons[team_seasons["Season"] == season]
    season_games = team_history[team_history["Season"] == season]
    champion = season_teams.loc[season_teams["Final Standing"].idxmin()]
    high = season_games.loc[season_games["Score"].idxmax()]
    cols = st.columns(4)
    cols[0].metric("Champion", champion["Manager"])
    cols[1].metric("Best record", f"{season_teams['Wins'].max()} wins")
    cols[2].metric("Highest week", f"{high['Score']:.2f}", high["Manager"])
    cols[3].metric(
        "Most active",
        season_teams.loc[season_teams["Acquisitions"].idxmax(), "Manager"],
    )

    st.subheader("Final standings")
    show_table(
        season_teams.sort_values("Final Standing")[
            [
                "Final Standing", "Manager", "Wins", "Losses",
                "Points For", "Points Against", "Acquisitions", "Trades",
            ]
        ]
    )
    if not player_history.empty:
        leaders = build_scoring_leaders_dataframe(
            player_history[player_history["Season"] == season]
        )
        st.subheader("Player scoring leaders")
        show_table(leaders.head(15))

with st.expander("Data notes"):
    st.write(
        f"League {LEAGUE_ID} is read through ESPN's unofficial Fantasy API. "
        "League history is presented by owner because fantasy team names may "
        "change between seasons. Older seasons or detailed transactions may "
        "have been removed by ESPN."
    )

report_html = archive_html_report(
    season_label=season_label,
    team_history=team_history,
    team_seasons=team_seasons,
    player_history=player_history,
)
st.download_button(
    "Download complete HTML report",
    data=report_html,
    file_name=f"phantasy-phootball-report-{season_label}.html",
    mime="text/html",
    width="stretch",
)
