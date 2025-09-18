"""Test purpose: verify that an **invalid subcommand** is rejected by the CLI,
exiting non-zero and showing a helpful error/usage. Supports CLIs that require
a FILE positional before the subcommand by retrying with a temp file.
"""

import re
import pytest

def test_invalid_subcommand_exits_nonzero_and_lists_choices(run_cli, tmp_path):
    # First try the simple form: no file, just a bogus command
    code, out, err = run_cli("definitely-not-a-cmd")
    text = (err or out or "")

    # If the CLI expects a file positional first, retry with a temp file
    needs_file = bool(re.search(
        r"(file .* required|the following arguments are required:.*file|file .* not found)",
        text, re.IGNORECASE
    ))
    if needs_file:
        dummy = tmp_path / "dummy.txt"
        dummy.write_text("dummy")  # ensure the file exists
        code, out, err = run_cli(str(dummy), "definitely-not-a-cmd")
        text = (err or out or "")

    text_l = text.lower()

    # Must exit with an error
    assert code in (1, 2), f"Expected non-zero exit for invalid subcommand, got {code}"

    # Be flexible across frameworks and messages
    helpful = any(
        phrase in text_l
        for phrase in (
            "invalid choice",     # argparse
            "unknown command",    # click/others
            "no such command",
            "choose from",
            "available commands",
        )
    ) or ("usage:" in text_l) or ("error:" in text_l)

    assert helpful, f"Expected a helpful error/usage, got: {text.strip() or '<empty>'}"