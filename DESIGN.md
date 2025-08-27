.
├─ backend/ # FastAPI service only (no business logic here)
│ ├─ app.py
│ ├─ requirements.txt
│ └─ Dockerfile
├─ src/ # domain logic used by backend + scripts
│ ├─ **init**.py
│ ├─ io_rsl.py # load/validate/normalize RSL
│ ├─ matrix.py # build/load distance/time matrices
│ ├─ model.py # build cuOpt problem (vehicles, orders, windows)
│ ├─ solve.py # baseline solve, warm-start insert, constraints
│ └─ analysis.py # deltas, candidate shortlist, KPIs
├─ scripts/
│ ├─ build_matrices.py # CLI wrapper → src.matrix
│ └─ demo_insert.py # CLI: baseline → insert new job → print report
├─ ui/
│ ├─ streamlit_app.py # calls backend endpoints; no solver logic here
│ ├─ requirements.txt
│ └─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ .gitignore
└─ README.md
