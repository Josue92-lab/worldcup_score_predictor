import pdfplumber
from src.config import SQUADS_PDF_PATH

def extract_first_page():
    print(f"Reading {SQUADS_PDF_PATH}...")
    if not SQUADS_PDF_PATH.exists():
        print("File not found!")
        return
    
    with pdfplumber.open(SQUADS_PDF_PATH) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        with open("first_page.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted to first_page.txt")

if __name__ == "__main__":
    extract_first_page()
