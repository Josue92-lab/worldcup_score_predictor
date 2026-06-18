import unicodedata
import yaml
from pathlib import Path
from rapidfuzz import process, fuzz
from src.config import TEAM_ALIASES_PATH

# Load aliases once
def load_aliases() -> dict:
    if TEAM_ALIASES_PATH.exists():
        with open(TEAM_ALIASES_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data and 'aliases' in data:
                return data['aliases']
    return {}

TEAM_ALIASES = load_aliases()
# Inverse mapping or just use aliases dict directly. 
# aliases format: "source_name": "Target Canonical Name"

def remove_accents(input_str: str) -> str:
    """Removes accents and special characters."""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def basic_normalize(name: str) -> str:
    """Strips whitespace, removes accents, and unifies casing for comparison, but preserves title case for display."""
    if not isinstance(name, str):
        return ""
    # We strip and title-case.
    # Note: For internal mapping, we might want to do lower-case comparison,
    # but the canonical name should look nice.
    clean_name = name.strip()
    # Normalize internal spaces
    clean_name = " ".join(clean_name.split())
    # Remove accents for a canonical mapping lookup
    clean_name = remove_accents(clean_name)
    return clean_name

def normalize_team_name(name: str, use_fuzzy: bool = False, known_canonical_names: list = None) -> str:
    """
    Normalizes a team name based on:
    1. Basic string cleaning
    2. Manual aliases in team_aliases.yml
    3. (Optional) Fuzzy matching to known canonical names
    """
    if not name:
        return ""
    
    clean_name = basic_normalize(name)
    
    # 1. Direct alias check
    if clean_name in TEAM_ALIASES:
        return TEAM_ALIASES[clean_name]
    
    # Check lowercase mapping just in case
    lower_mapping = {k.lower(): v for k, v in TEAM_ALIASES.items()}
    if clean_name.lower() in lower_mapping:
        return lower_mapping[clean_name.lower()]
        
    # 2. Fuzzy matching (if enabled and canonical list provided)
    if use_fuzzy and known_canonical_names:
        # Rapidfuzz process.extractOne
        match = process.extractOne(clean_name, known_canonical_names, scorer=fuzz.WRatio)
        if match:
            best_match, score, _ = match
            if score >= 90.0:  # High confidence threshold
                return best_match
                
    # If no alias, return the cleaned name
    return clean_name
