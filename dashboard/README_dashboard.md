# Interactive dashboard for the reviews / products / users CSVs

## Files
- `dashboard_app.py`
- `requirements_dashboard.txt`

## Expected CSV names
The app auto-loads these files if they are in the same folder as the app:
- `products_clean.csv`
- `reviews_clean_no_exact_duplicates.csv`
- `user_summary.csv`

If the CSVs are somewhere else, start the app and upload them from the sidebar.

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_dashboard.txt
streamlit run dashboard_app.py
```

## What is inside
- Overview KPIs
- Filters by year, rating, verified purchase, helpful votes, and text presence
- Review trends and distributions
- Product charts, metadata coverage, store leaderboard, and searchable product table
- User concentration analysis including top 1% / top 5% review share
- Data dictionary / preview tab

## Notes
- The app reads the full CSVs directly, so it stays interactive on the real dataset.
- To keep memory use reasonable, it loads only the columns needed for the dashboard.
