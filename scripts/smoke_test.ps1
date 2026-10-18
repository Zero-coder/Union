$ErrorActionPreference = "Stop"
$unionRepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $unionRepoRoot
try {
    # Functional tests use small controlled fixtures, without benchmark downloads.
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
