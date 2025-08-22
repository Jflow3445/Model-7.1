# run_training.ps1

# ── Allow this script to run ───────────────────────────────────────────
Set-ExecutionPolicy Bypass -Scope Process -Force

# ── Change into your project directory ─────────────────────────────────
Set-Location -LiteralPath 'C:\Users\DummYBoY\Documents\Model 4'

# ── Build the inner command (activate + run) ────────────────────────────
$inner = '. .\.venv\Scripts\Activate.ps1; python main.py'

# ── Launch a new PS window, keep it open so you can watch logs ─────────
Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoExit',
    '-ExecutionPolicy', 'Bypass',
    '-Command', $inner
) -WindowStyle Normal
