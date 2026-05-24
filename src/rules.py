import json
from functools import lru_cache
from pathlib import Path

# helper functions for turning classfication to disposal recommendations

# config file for disposal rules
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "disposal_rules.json"
DEFAULT_LOCAL_RULES_NOTE = "Local rules vary. Check your city or county guidance."

# Load disposal rules from a JSON configuration file.
@lru_cache(maxsize=1)
def load_disposal_rules(config_path: str | Path = DEFAULT_RULES_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Get a disposal recommendation for a given material classification category.
def get_disposal_recommendation(category: str, config_path: str | Path = DEFAULT_RULES_PATH) -> dict:
    """Map a predicted material category to a configured disposal action."""
    normalized = category.strip().lower()
    rules = load_disposal_rules(config_path)
    rule = rules.get(
        normalized,
        {
            "decision": "Check local rules",
            "explanation": "This category is not in the current disposal config.",
        },
    )

    return {
        "decision": rule["decision"],
        "explanation": rule.get("explanation", ""),
        "local_rules_note": DEFAULT_LOCAL_RULES_NOTE,
    }
