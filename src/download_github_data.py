import requests
from src.config import RAW_DIR

GITHUB_BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/"
FILES_TO_DOWNLOAD = [
    "results.csv",
    "shootouts.csv",
    "goalscorers.csv"
]

def download_github_data():
    target_dir = RAW_DIR / "international_results"
    target_dir.mkdir(exist_ok=True)
    
    print("Downloading historical international results from GitHub...")
    for filename in FILES_TO_DOWNLOAD:
        url = GITHUB_BASE_URL + filename
        print(f"Fetching {url}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            file_path = target_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved {filename} to {file_path}")
        else:
            print(f"Failed to download {filename} (HTTP {response.status_code})")

if __name__ == "__main__":
    download_github_data()
