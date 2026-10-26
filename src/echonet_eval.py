"""Refuse EchoNet evaluation when approved local data is absent."""
from pathlib import Path
import json
import os

def main() -> None:
    """Check only approved local dataset locations."""
    candidates = [Path(os.environ["ECHONET_DIR"])] if os.environ.get("ECHONET_DIR") else []
    candidates.append(Path("data"))
    found = next((path for path in candidates if (path / "FileList.csv").exists() and (path / "VolumeTracings.csv").exists()), None)
    if found is None:
        print("NOT RUN: dataset not found")
        return
    print(f"Dataset found at {found}. Read CSV headers before evaluation.")

if __name__ == "__main__": main()
