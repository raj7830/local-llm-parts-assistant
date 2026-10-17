"""Tools the LLM can call. Docstrings double as the tool schema sent to the model
(the ollama client builds the JSON schema from type hints + Google-style docstrings)."""

from __future__ import annotations

import json
from functools import lru_cache

from src.parts import PartsSimilarity, load_parts
from src.timediff import business_hours_between as _business_hours
from src.timediff import hours_between as _hours


@lru_cache(maxsize=1)
def parts_model() -> PartsSimilarity:
    return PartsSimilarity().fit(load_parts())


def _records(df) -> list[dict]:
    df = df.rename(columns={"description_source": "matched_on"})
    df["DESCRIPTION"] = df["DESCRIPTION"].fillna("(no description; matched on structured attributes)")
    return df.to_dict(orient="records")


def find_alternative_parts(part_id: str, k: int = 5) -> str:
    """Find the most similar alternative parts for an existing part ID from the parts catalogue.

    Args:
      part_id: Part identifier such as "A12" (letter A followed by a number).
      k: Number of alternatives to return (default 5).

    Returns:
      JSON with the part's own description and a ranked list of alternatives with similarity scores (0-1).
    """
    model = parts_model()
    pid = part_id.strip().upper()
    if pid not in model.index_of:
        return json.dumps({"error": f"Part {part_id!r} does not exist. IDs range from A1 to A{len(model.parts)}."})
    row = model.parts.loc[model.index_of[pid]]
    return json.dumps(
        {
            "part_id": pid,
            "description": row["DESCRIPTION"] if isinstance(row["DESCRIPTION"], str) else None,
            "alternatives": _records(model.alternatives(pid, int(k))),
        }
    )


def search_parts(description: str, k: int = 5) -> str:
    """Search the parts catalogue by a free-text description of the required part.

    Args:
      description: Wanted properties, e.g. "time lag 2A 250V 5x20mm glass".
      k: Number of results to return (default 5).

    Returns:
      JSON list of the best matching parts with similarity scores (0-1).
    """
    return json.dumps({"results": _records(parts_model().search(description, int(k)))})


def hours_between(start: str, end: str) -> str:
    """Rounded number of full hours between two timestamps.

    Args:
      start: Timestamp in the format "YYYY/MM/DD HH:MM", e.g. "2022/02/15 00:05".
      end: Timestamp in the format "YYYY/MM/DD HH:MM".

    Returns:
      JSON with the number of hours.
    """
    try:
        return json.dumps({"hours": _hours(start, end)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


def business_hours_between(start: str, end: str) -> str:
    """Rounded number of working hours between two timestamps, counting only 09:00-17:00 on weekdays (Mon-Fri).

    Args:
      start: Timestamp in the format "YYYY/MM/DD HH:MM".
      end: Timestamp in the format "YYYY/MM/DD HH:MM".

    Returns:
      JSON with the number of working hours.
    """
    try:
        return json.dumps({"business_hours": _business_hours(start, end)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


TOOLS = {f.__name__: f for f in (find_alternative_parts, search_parts, hours_between, business_hours_between)}
