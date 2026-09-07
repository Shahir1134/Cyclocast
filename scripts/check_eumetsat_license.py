"""
scripts/check_eumetsat_license.py
==================================
Checks if EUMETSAT licence/API access is working by performing a real,
small download of the latest available pass using the eumdac CLI
(the officially supported download path), rather than eumdac's Python
product.open()/stream API, which has been found to throw spurious
403 Unauthorised errors on some collections even when the account,
licence, and credentials are all valid.
"""

import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

# Ensure project root is available
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLLECTION = "EO:EUM:DAT:MSG:HRSEVIRI-IODC"


def check_status():
    eumdac_path = shutil.which("eumdac")
    if not eumdac_path:
        print("[ERROR] 'eumdac' CLI not found on PATH. Is it installed (pip install eumdac)?")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[INFO] Attempting a real download of the latest {COLLECTION} pass...")
        result = subprocess.run(
            [
                eumdac_path,
                "download",
                "-c", COLLECTION,
                "--limit", "1",
                "-o", tmpdir,
            ],
            capture_output=True,
            text=True,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        downloaded_files = list(Path(tmpdir).rglob("*"))
        downloaded_files = [f for f in downloaded_files if f.is_file()]

        if result.returncode == 0 and downloaded_files:
            total_bytes = sum(f.stat().st_size for f in downloaded_files)
            print("\n" + "=" * 60)
            print("  SUCCESS! LICENCE AND API ACCESS ARE ACTIVE AND WORKING!")
            print("=" * 60)
            print(f"Downloaded {len(downloaded_files)} file(s), {total_bytes / 1e6:.1f} MB total.")
            print("You can now run: python scripts/eumetsat_live_pipeline.py --dry-run")
            return True
        else:
            print("\n" + "-" * 60)
            print("  STATUS: DOWNLOAD FAILED")
            print("-" * 60)
            print(f"Exit code: {result.returncode}")
            if stdout.strip():
                print(f"stdout:\n{stdout}")
            if stderr.strip():
                print(f"stderr:\n{stderr}")
            print(
                "\nIf this mentions 'Unauthorised' or '403', double-check your licence at "
                "https://user.eumetsat.int/profile?activeTab=data-licenses and that your "
                "credentials are set via: eumdac set-credentials <key> <secret>"
            )
            return False


if __name__ == "__main__":
    check_status()