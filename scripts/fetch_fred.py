"""Fetch daily Treasury rates from FRED via CSV endpoint."""
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

SERIES = {
    "DGS3MO": "t_bill_3mo", "DGS1": "t_note_1y", "DGS10": "t_bond_10y",
    "VIXCLS": "vix", "VXVCLS": "vix_3m",
    "BAMLH0A0HYM2": "hy_spread",
    "DTWEXBGS": "usd_index",
}
MONTHS_BACK = 63
RAW = Path(__file__).parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

end = datetime.utcnow().date()
start = end - timedelta(days=MONTHS_BACK * 31)

frames = []
for code, name in SERIES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start}&coed={end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    s = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"], index_col="observation_date")
    s = s.rename(columns={code: name}).replace(".", pd.NA).astype(float)
    frames.append(s)

df = frames[0].join(frames[1:], how="outer").sort_index()
out = RAW / "macro_daily.parquet"
df.to_parquet(out)
print(f"rates: {len(df):,} rows, cols={list(df.columns)}, range={df.index.min().date()}..{df.index.max().date()} -> {out}")
