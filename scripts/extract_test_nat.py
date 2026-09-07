import zipfile
from pathlib import Path

zip_path = Path("test_download/MSG2-SEVI-MSG15-0100-NA-20260905052739.278000000Z-NA.zip")
out_dir = Path("tmp_eumetsat")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Unzipping {zip_path} to {out_dir}...")
with zipfile.ZipFile(zip_path, 'r') as z:
    for name in z.namelist():
        if name.endswith(".nat"):
            target_nat = out_dir / name
            if not target_nat.exists():
                print(f"Extracting {name}...")
                z.extract(name, out_dir)
            else:
                print(f"Already extracted: {target_nat} ({target_nat.stat().st_size / 1e6:.1f} MB)")
            print("Extracted nat path:", target_nat)
