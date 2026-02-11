# app.py
import os
import logging
from logging import handlers
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path

# =========================
# Initialize FastAPI app
# =========================
app = FastAPI(title="Smart Search API", version="1.1.0")

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sessions (if you don't need sessions, remove this line and the itsdangerous dep)
app.add_middleware(SessionMiddleware, secret_key="super-secret-key")


# =========================
# Configure Logging
# =========================
cwd = os.getcwd()
log_dir = os.path.join(cwd, "log")
print(log_dir)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "app.log")
log_handler = handlers.TimedRotatingFileHandler(log_file, when="midnight")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")
log_handler.setFormatter(formatter)

logger = logging.getLogger("smart_search")
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)


# =========================
# Load Data
# =========================
data_dir = os.path.join(cwd, "data")
BASE_DIR = Path(__file__).resolve().parent  # /home/cdn/smart_search_app/app
data_dir = BASE_DIR / "data"
print(data_dir)

def _read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, encoding="latin1")

try:
    pincodes_df = _read_csv_safely(os.path.join(data_dir, "pincodes.csv"))
    schools_df  = _read_csv_safely(os.path.join(data_dir, "schools.csv"))
    colleges_df = _read_csv_safely(os.path.join(data_dir, "colleges.csv"))
    cities_df = _read_csv_safely(os.path.join(data_dir, "cities.csv"))
    logger.info(
        "Loaded CSVs: pincodes(%s rows), schools(%s rows), colleges(%s rows), cities(%s rows)",
        len(pincodes_df), len(schools_df), len(colleges_df), len(cities_df)
    )
except Exception as e:
    logger.exception("Failed to load CSVs: %s", e)
    raise


# =========================
# Helpers
# =========================
def df_to_records_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to JSON-safe records."""
    df = df.replace([np.inf, -np.inf], pd.NA)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Required columns missing in dataset: {missing}"
        )


def search_in_dataframe(
    df: pd.DataFrame, column: str, query: str, case_sensitive: bool, startwith: bool
) -> List[Dict[str, Any]]:
    require_columns(df, [column])
    series = df[column].astype(str)

    if not case_sensitive:
        series = series.str.lower()
        query = query.lower()

    if startwith:
        matched = df.loc[series.str.startswith(query, na=False)]
    else:
        matched = df.loc[series.str.contains(query, na=False)]

    return df_to_records_safe(matched)


def search_pincode(query: str, case_sensitive: bool, startwith: bool) -> List[Dict[str, Any]]:
    return search_in_dataframe(pincodes_df, "PinCode", query, case_sensitive, startwith)

def search_schools(query: str, case_sensitive: bool, startwith: bool) -> List[Dict[str, Any]]:
    return search_in_dataframe(schools_df, "School Name", query, case_sensitive, startwith)

def search_colleges(query: str, case_sensitive: bool, startwith: bool) -> List[Dict[str, Any]]:
    return search_in_dataframe(colleges_df, "College", query, case_sensitive, startwith)

def search_cities(query: str, case_sensitive: bool, startwith: bool) -> List[Dict[str, Any]]:
    return search_in_dataframe(cities_df, "City", query, case_sensitive, startwith)


# =========================
# Routes
# =========================
@app.get("/search/{action_type}")
async def search(
    action_type: str,
    query: str = Query(..., description="Search text to match"),
    case_sensitive: bool = Query(False, description="Match case (default: False)"),
    startwith: bool = Query(False, description="Match only if startswith (default: False)"),
):
    try:
        atype = action_type.strip().lower()
        if not query:
            raise HTTPException(status_code=400, detail="Missing required parameter: query")

        if atype == "pincode":
            results = search_pincode(query, case_sensitive, startwith)
        elif atype == "school":
            results = search_schools(query, case_sensitive, startwith)
        elif atype == "college":
            results = search_colleges(query, case_sensitive, startwith)
        elif atype == "city":
            results = search_cities(query, case_sensitive, startwith)
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid action_type. Use one of: pincode, school, college",
            )

        logger.info(
            "Search type=%s query='%s' case_sensitive=%s startwith=%s count=%d",
            atype, query, case_sensitive, startwith, len(results)
        )
        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /search/%s: %s", action_type, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"status": "ok", "message": "Smart Search API is running"}


# =========================
# Local run
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5600, reload=True)
