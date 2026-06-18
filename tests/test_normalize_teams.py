import pytest
from src.normalize_teams import remove_accents, basic_normalize, normalize_team_name

def test_remove_accents():
    assert remove_accents("Curaçao") == "Curacao"
    assert remove_accents("Türkiye") == "Turkiye"
    assert remove_accents("Côte d'Ivoire") == "Cote d'Ivoire"

def test_basic_normalize():
    assert basic_normalize("  Bosnia   & Herzegovina  ") == "Bosnia & Herzegovina"

def test_normalize_with_aliases():
    # Curacao variants should all map to "Curaçao"
    assert normalize_team_name("Curacao") == "Curaçao"
    assert normalize_team_name("Curaçao") == "Curaçao"
    
    # Ivory Coast variants
    assert normalize_team_name("Ivory Coast") == "Côte D'Ivoire"
    assert normalize_team_name("Cote D'Ivoire") == "Côte D'Ivoire"
    assert normalize_team_name("Côte D'Ivoire") == "Côte D'Ivoire"

    # Turkey variants
    assert normalize_team_name("Turkey") == "Türkiye"
    assert normalize_team_name("Turkiye") == "Türkiye"
    assert normalize_team_name("Türkiye") == "Türkiye"

    # South Korea → Korea Republic
    assert normalize_team_name("South Korea") == "Korea Republic"
    
    # USA → United States
    assert normalize_team_name("USA") == "United States"
