from __future__ import annotations

import argparse
from pathlib import Path

from cbpe.pipeline import reproduce
from cbpe.verify import verify_manifest


ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="cbpe")
    commands = root.add_subparsers(dest="command", required=True)
    reproduce_parser = commands.add_parser("reproduce", help="Run the auditable reproduction pipeline")
    reproduce_parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    verify_parser = commands.add_parser("verify", help="Verify a reproduction manifest")
    verify_parser.add_argument("manifest", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "reproduce":
        manifest = reproduce(ROOT, args.profile)
        print(f"Reproduction complete: {manifest.relative_to(ROOT)}")
        return
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    errors = verify_manifest(ROOT, manifest_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Manifest verified: {manifest_path.relative_to(ROOT)}")

