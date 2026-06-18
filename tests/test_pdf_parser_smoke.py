import pytest
from src.parse_squads_pdf import parse_squads_pdf
from src.config import SQUADS_PDF_PATH

@pytest.mark.skipif(not SQUADS_PDF_PATH.exists(), reason="PDF file not found")
def test_pdf_parser_smoke():
    # Attempt to parse
    df = parse_squads_pdf()
    
    # We expect some data back
    # Just a smoke test, so we ensure it's a dataframe and has expected columns
    assert df is not None
    assert 'team' in df.columns
    assert 'player_name' in df.columns
    assert 'position' in df.columns
    
    if not df.empty:
        # Check that we extracted some players
        assert len(df) > 10
