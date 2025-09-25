import argparse
import subprocess
import sys

def build_parser():
    p = argparse.ArgumentParser(prog="run", description="Team CLI")
    subs = p.add_subparsers(dest="command", required=True)

    # --- test subcommand ---
    test_p = subs.add_parser("test", help="Run pytest suite")
    test_p.add_argument(
        "paths",
        nargs="*",
        default=["tests"],
        help="Files/dirs to test (default: tests)",
    )
    test_p.add_argument("-q", "--quiet", action="store_true", help="Quiet pytest output")
    test_p.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output (-v)")

    # Coverage options (pytest-cov)
    test_p.add_argument(
        "--cov",
        nargs="?",               # allows --cov or --cov=PACKAGE/PATH
        const="src",             # if provided without value, default to 'src'
        help="Enable coverage for the given package/path (default: src).",
    )
    test_p.add_argument(
        "--cov-branch",
        action="store_true",
        help="Measure branch coverage.",
    )
    test_p.add_argument(
        "--cov-report",
        default="term-missing",
        help="Coverage report type (e.g., term, term-missing, html). Default: term-missing.",
    )
    test_p.add_argument(
        "--fail-under",
        dest="cov_fail_under",
        type=float,
        help="Fail if total coverage percentage is below this value.",
    )

    # Pass-through for any additional pytest args after --
    test_p.add_argument(
        "--",
        dest="extra",
        nargs=argparse.REMAINDER,
        help="Pass any remaining args straight to pytest (use after --).",
    )

    return p

def _pytest_cmd_from_args(args) -> list[str]:
    cmd = [sys.executable, "-m", "pytest"]

    # verbosity
    if args.quiet:
        cmd.append("-q")
    if args.verbose:
        cmd.append("-v")

    # coverage
    if args.cov:
        cmd += ["--cov", args.cov, "--cov-report", args.cov_report]
        if args.cov_branch:
            cmd.append("--cov-branch")
        if args.cov_fail_under is not None:
            cmd += ["--cov-fail-under", str(args.cov_fail_under)]

    # targets
    cmd += args.paths

    # raw extras (after --)
    if args.extra:
        cmd += args.extra

    return cmd

def cmd_test(args) -> int:
    cmd = _pytest_cmd_from_args(args)
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode

# This module-level run() lets Orchestrator (or others) call tests directly.
# You can pass pytest_args like ["--cov=src", "--cov-report=term-missing"].
def run(paths=None, pytest_args=None) -> int:
    """Run pytest and return its exit code."""
    if paths is None:
        paths = ["tests"]
    if pytest_args is None:
        pytest_args = ["-q", "-rs"]  # quiet + show skip reasons

    cmd = [sys.executable, "-m", "pytest", *pytest_args, *paths]
    # print("Running:", " ".join(cmd))
    # Suppress pytest output by redirecting stdout and stderr to DEVNULL
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "test":
        return cmd_test(args)

    # elif args.command == "install": ...
    return 0

if __name__ == "__main__":
    raise SystemExit(main())