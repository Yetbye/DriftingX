import argparse

from utils.misc import stamp_workdir


def _emit_report(workdir: str) -> None:
    """Render the post-hoc report for this run.

    Runs on every exit path (including after a crash) so an offline cluster job
    always leaves a human-readable summary without anyone watching it live.
    Best-effort: never let reporting break or mask the training outcome.
    """
    import sys
    from pathlib import Path

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
        from analyze_run import analyze_run

        print("[report] writing log/report.png and log/report.md ...", flush=True)
        analyze_run(Path(workdir))
    except Exception as exc:  # noqa: BLE001
        print(f"[report] skipped ({type(exc).__name__}: {exc})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--gen", action="store_true", help="Run generator training loop. Default runs MAE training.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling.")
    parser.add_argument("--no-stamp", action="store_true",
                        help="Use --workdir verbatim (no timestamp). Needed for resumable/offline runs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing run: --workdir must already hold a checkpoint.")
    args = parser.parse_args()
    args.workdir = stamp_workdir(args.workdir, no_stamp=args.no_stamp, resume=args.resume)
    args.output_dir = args.workdir

    try:
        if args.gen:
            from train import main as train_gen_main

            train_gen_main(args)
        else:
            from train_mae import main as train_mae_main

            train_mae_main(args)
    finally:
        # Emit the report whether training finished or raised, so the offline
        # run directory is always self-describing.
        _emit_report(args.workdir)


if __name__ == "__main__":
    main()
