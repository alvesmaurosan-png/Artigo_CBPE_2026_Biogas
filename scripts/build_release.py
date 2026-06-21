from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "results" / "release"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    archive = RELEASE_DIR / "cbpe-2026-full-results.zip"
    candidates = [ROOT / "results" / "paper", ROOT / "results" / "runs"]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as output:
        for directory in candidates:
            if directory.exists():
                for path in directory.rglob("*"):
                    if path.is_file():
                        output.write(path, path.relative_to(ROOT))
    checksum = {"file": archive.name, "sha256": sha256(archive), "bytes": archive.stat().st_size}
    (RELEASE_DIR / "checksums.json").write_text(json.dumps(checksum, indent=2) + "\n", encoding="utf-8")
    print(archive)


if __name__ == "__main__":
    main()

