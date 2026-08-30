#!/usr/bin/env python3
"""Scrape the FlavorDB entity-molecule graph.

The paper seeds Chem and Core from "an 80,019-edge typed FlavorDB
ingredient-compound graph, 2,247 typed compound nodes across 15 categories".
FlavorDB has no bulk export, but its per-entity JSON endpoint is open:

    https://cosylab.iiitd.edu.in/flavordb2/entities_json?id=N

Each entity carries a food category and a list of molecules; each molecule
carries PubChem identity plus functional groups and flavour descriptors. We pull
every entity id and cache the raw JSON so the graph can be rebuilt offline
without re-hitting the server.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "flavordb"
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "entities"
RAW.mkdir(exist_ok=True)

URL = "https://cosylab.iiitd.edu.in/flavordb2/entities_json?id={}"
MAX_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1200


def fetch(eid: int) -> tuple[int, str]:
    dest = RAW / f"{eid}.json"
    if dest.exists() and dest.stat().st_size > 40:
        return eid, "cached"
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(eid), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                body = r.read()
            if len(body) < 40:
                return eid, "empty"
            json.loads(body)
            dest.write_bytes(body)
            return eid, "ok"
        except urllib.error.HTTPError as e:
            if e.code in (404, 500):
                return eid, f"http{e.code}"
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return eid, "fail"


def main() -> None:
    ids = list(range(1, MAX_ID + 1))
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(8) as ex:
        for n, (eid, status) in enumerate(ex.map(fetch, ids), 1):
            stats[status] = stats.get(status, 0) + 1
            if n % 100 == 0:
                print(f"  {n}/{len(ids)}  {stats}", flush=True)
    print("done:", stats)
    print("entity files:", len(list(RAW.glob('*.json'))))


if __name__ == "__main__":
    main()
