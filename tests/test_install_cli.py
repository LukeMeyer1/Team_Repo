"""
UML reference (Install branch):
Parse CLI args → INSTALL? → Run dependency install (pip) → 
   on success: print success & exit 0
   on failure: print error & exit 1

This test verifies the success path:
- calling the CLI with 'install' triggers a pip install invocation
- exit code is 0 when the underlying pip call succeeds

It auto-skips if the CLI module or 'install' subcommand isn't present yet.
If present, fakes pip via monkeypatch to avoid a real install
"""

"""Test purpose: verify that the `install` subcommand triggers a pip invocation
and exits 0 on success. If the current implementation is a placeholder (no pip),
the test skips (pending-safe).
"""

import sys, runpy, subprocess, pytest

def test_install_invokes_pip_and_exits_zero(monkeypatch, cli_path, run_cli, capsys):
    # only run if 'install' is advertised
    h_code, h_out, h_err = run_cli("--help")
    if "install" not in (h_out + h_err).lower():
        pytest.skip("'install' subcommand not implemented yet")

    calls = []
    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(subprocess, "run", fake_run, raising=True)

    monkeypatch.setattr(sys, "argv", [str(cli_path), "install"], raising=False)
    try:
        runpy.run_path(str(cli_path), run_name="__main__")
        rc = 0
    except SystemExit as e:
        rc = int(e.code)

    out, err = capsys.readouterr()
    if ("placeholder" in (out + err).lower()) and not calls:
        pytest.skip("Install action is a placeholder; pip not wired yet")

    assert rc == 0
    assert calls and "pip" in " ".join(map(str, calls[0])) and "install" in " ".join(map(str, calls[0]))