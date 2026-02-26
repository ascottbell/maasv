"""CLI entry point for maasv-ingest."""

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maasv-ingest",
        description="Ingest text, files, or directories into maasv memory.",
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("path", nargs="?", type=Path, help="File or directory to ingest")
    input_group.add_argument("--text", type=str, help="Raw text to ingest")
    input_group.add_argument("--stdin", action="store_true", help="Read text from stdin")

    # Metadata
    parser.add_argument("--category", default="imported", help="Memory category (default: imported)")
    parser.add_argument("--source", default=None, help="Source label (default: filename)")
    parser.add_argument("--origin", default=None, help="System origin (e.g., chatgpt, notion)")
    parser.add_argument("--origin-interface", default=None, help="Interface (e.g., api, cli)")

    # Behavior
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk without storing")
    parser.add_argument("--no-extraction", action="store_true", help="Skip LLM entity extraction")
    parser.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirectories")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Target chunk size in chars (default: 3000)")

    # Output
    parser.add_argument("--quiet", action="store_true", help="Only print final summary")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    return parser


def _progress_printer(quiet: bool):
    """Return a progress callback that prints to stderr."""
    if quiet:
        return None

    def callback(index: int, message: str):
        print(f"  {message}", file=sys.stderr)

    return callback


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Initialize maasv (reuses same MAASV_* env vars as maasv-server)
    if not args.dry_run:
        try:
            from maasv.server.main import _init_maasv

            _init_maasv()
        except Exception as e:
            print(f"Failed to initialize maasv: {e}", file=sys.stderr)
            print("Check MAASV_* environment variables.", file=sys.stderr)
            sys.exit(1)

    progress = _progress_printer(args.quiet)
    run_extraction = not args.no_extraction

    try:
        if args.text:
            from maasv.ingestion.pipeline import ingest_text

            if not args.quiet:
                print("Ingesting text...", file=sys.stderr)

            report = ingest_text(
                text=args.text,
                category=args.category,
                source=args.source or "cli",
                origin=args.origin,
                origin_interface=args.origin_interface or "cli",
                dry_run=args.dry_run,
                run_extraction=run_extraction,
                chunk_size=args.chunk_size,
                progress_callback=progress,
            )

        elif args.stdin:
            from maasv.ingestion.pipeline import ingest_text

            text = sys.stdin.read()
            if not text.strip():
                print("No input received from stdin.", file=sys.stderr)
                sys.exit(1)

            if not args.quiet:
                print("Ingesting from stdin...", file=sys.stderr)

            report = ingest_text(
                text=text,
                category=args.category,
                source=args.source or "stdin",
                origin=args.origin,
                origin_interface=args.origin_interface or "cli",
                dry_run=args.dry_run,
                run_extraction=run_extraction,
                chunk_size=args.chunk_size,
                progress_callback=progress,
            )

        elif args.path.is_dir():
            from maasv.ingestion.pipeline import ingest_directory

            if not args.quiet:
                print(f"Ingesting directory: {args.path}", file=sys.stderr)

            report = ingest_directory(
                path=args.path,
                category=args.category,
                origin=args.origin,
                origin_interface=args.origin_interface or "cli",
                recursive=not args.no_recursive,
                dry_run=args.dry_run,
                run_extraction=run_extraction,
                chunk_size=args.chunk_size,
                progress_callback=progress,
            )

        elif args.path.is_file():
            from maasv.ingestion.pipeline import ingest_file

            if not args.quiet:
                print(f"Ingesting file: {args.path}", file=sys.stderr)

            report = ingest_file(
                path=args.path,
                category=args.category,
                source=args.source,
                origin=args.origin,
                origin_interface=args.origin_interface or "cli",
                dry_run=args.dry_run,
                run_extraction=run_extraction,
                chunk_size=args.chunk_size,
                progress_callback=progress,
            )

        else:
            print(f"Path not found: {args.path}", file=sys.stderr)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted. Already-committed data is preserved.", file=sys.stderr)
        sys.exit(130)

    # Print report
    if args.dry_run and not args.quiet:
        print("\n[DRY RUN — nothing was stored]", file=sys.stderr)

    print(report.summary())


if __name__ == "__main__":
    main()
