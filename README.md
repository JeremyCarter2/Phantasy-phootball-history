# Phantasy Phootball History

A private-league history dashboard for ESPN league `682600`. It includes:

- Weekly team and player leaderboards
- Player season scoring leaders using each season's league settings
- All-time owner records, championships, and playoff appearances
- All-play records and schedule-luck estimates
- Head-to-head rivalry records
- League single-week and matchup records
- Actual-versus-optimal lineup efficiency
- Transaction activity summaries
- Inferred trade winner/loser analysis based on post-trade fantasy points
- Player roster history
- Season recap pages
- CSV exports for every table and a complete downloadable HTML report

Archive seasons are cached for 24 hours and loaded concurrently to reduce
wait times. Player views still take longer because ESPN requires one box-score
request per week.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
.venv/bin/streamlit run app.py
```

Put the league owner's `espn_s2` and `SWID` cookies in
`.streamlit/secrets.toml`. Never commit that file.

## Deploy

1. Push the project to a private or public GitHub repository.
2. Create an app on [Streamlit Community Cloud](https://share.streamlit.io/).
3. Set the entry point to `app.py`.
4. In the app's **Settings > Secrets**, add:

```toml
ESPN_S2 = "your-espn-s2-cookie"
SWID = "{your-swid-cookie}"
```

The credentials remain server-side and are not displayed to visitors.

## Notes

ESPN's Fantasy API is unofficial. ESPN may change endpoints or remove old
season data without notice. The dashboard skips unavailable seasons and
continues loading the rest of the requested range. Detailed historical
transaction events are not consistently retained, so the transaction view
uses ESPN's official team-level season counters. Trade analysis identifies
reciprocal week-to-week roster moves and labels those deals as inferred.
