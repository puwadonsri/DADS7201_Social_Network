# DADS7201 — Social Network Analysis

Coursework for **DADS7201 — Social Network Analysis** at NIDA, organised
by week. Each week is a self-contained Streamlit application with its own
data, dependencies, and deploy configuration.

## Weeks

| Week | Topic | Folder | Status |
|---|---|---|---|
| 1 | Bipartite network: SET50 companies ↔ top-5 major shareholders | [`Week1/`](Week1/) | ✅ deployed |
| 2 | _(coming soon)_ | `Week2/` | — |
| 3 | _(coming soon)_ | `Week3/` | — |

## How this repo is organised

Each `WeekN/` folder is **independent**:

- has its own `streamlit_app.py`, `requirements.txt`, `packages.txt`
- can be deployed as a separate Streamlit Cloud app pointing at
  `WeekN/streamlit_app.py`
- shares no code with other weeks — copy/paste any patterns you want
  to reuse

## Adding a new week

```powershell
# from the repo root
mkdir Week2
# copy the Week1 skeleton as a starting point if useful
Copy-Item -Recurse Week1\.streamlit, Week1\requirements.txt, Week1\packages.txt Week2\
# build your app in Week2/streamlit_app.py
git add Week2
git commit -m "Week2: initial app"
git push
```

Then on https://share.streamlit.io: **New app** → repo
`puwadonsri/DADS7201_Social_Network`, main file `Week2/streamlit_app.py`.

## Author

Puwadon Sri ([@puwadonsri](https://github.com/puwadonsri)) ·
DADS, NIDA · 2026
