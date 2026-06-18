import re
import pdfplumber
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SQUADS_PDF_PATH, PROCESSED_DIR
from src.normalize_teams import normalize_team_name

def parse_squads_pdf() -> pd.DataFrame:
    if not SQUADS_PDF_PATH.exists():
        print(f"Error: {SQUADS_PDF_PATH} not found.")
        return pd.DataFrame()
        
    print(f"Parsing {SQUADS_PDF_PATH}...")
    
    records = []
    
    # Team header regex: "Algeria (ALG)"
    team_re = re.compile(r'^([A-Za-z\s&\'\-çôüÇÔÜ]+)\s+\([A-Z]{3}\)$')
    # Player line regex: 1 PO Name Name Name 19/2/2000 FC Club (SUI) 194 2 0
    player_re = re.compile(r'^(\d+)\s+([A-Z]{2})\s+(.+?)\s+(\d{1,2}/\d{1,2}/\d{4})\s+(.+?\([A-Z]{3}\))\s+(\d+)\s+(\d+)\s+(\d+)$')
    
    current_team = None
    
    with pdfplumber.open(SQUADS_PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for team header
                team_match = team_re.match(line)
                if team_match:
                    current_team = team_match.group(1).strip()
                    # We could extract FIFA code but not strictly required
                    continue
                
                # Check for player
                player_match = player_re.match(line)
                if player_match and current_team:
                    pos = player_match.group(2)
                    names_chunk = player_match.group(3)
                    dob = player_match.group(4)
                    club = player_match.group(5)
                    height = int(player_match.group(6))
                    caps = int(player_match.group(7))
                    goals = int(player_match.group(8))
                    
                    # Heuristic to get shirt name: just take the whole names chunk for now
                    # or the last word.
                    names_parts = names_chunk.split()
                    shirt_name = names_parts[-1] if len(names_parts) > 0 else names_chunk
                    
                    records.append({
                        'team': normalize_team_name(current_team),
                        'raw_team': current_team,
                        'player_name': names_chunk,
                        'shirt_name': shirt_name,
                        'position': pos,
                        'dob': dob,
                        'club': club,
                        'height': height,
                        'caps': caps,
                        'goals': goals
                    })

    df = pd.DataFrame(records)
    
    if not df.empty:
        out_path = PROCESSED_DIR / "squads_parsed.csv"
        df.to_csv(out_path, index=False)
        print(f"Parsed {len(df)} players from {len(df['team'].unique())} teams.")
        print(f"Saved to {out_path}")
    else:
        print("Failed to parse any players.")
        
    return df

if __name__ == "__main__":
    df = parse_squads_pdf()
