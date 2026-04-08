param(
    [ValidateSet("setup", "doctor", "train", "eval", "run", "test", "lint")]
    [string]$Task = "run"
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

switch ($Task) {
    "setup" {
        Invoke-Checked @($python, "-m", "pip", "install", "-U", "pip")
        Invoke-Checked @($python, "-m", "pip", "install", "-e", ".[dev]")
    }
    "doctor" {
        Invoke-Checked @($python, "-m", "afrorad_pipeline.doctor")
    }
    "train" {
        Invoke-Checked @($python, "-m", "accelerate.commands.launch", "-m", "afrorad_pipeline.train")
    }
    "eval" {
        Invoke-Checked @($python, "-m", "accelerate.commands.launch", "-m", "afrorad_pipeline.eval")
    }
    "run" {
        Invoke-Checked @($python, "-m", "afrorad_pipeline.doctor")
        Invoke-Checked @($python, "-m", "accelerate.commands.launch", "-m", "afrorad_pipeline.train")
        Invoke-Checked @($python, "-m", "accelerate.commands.launch", "-m", "afrorad_pipeline.eval")
    }
    "test" {
        Invoke-Checked @($python, "-m", "pytest", "-q")
    }
    "lint" {
        Invoke-Checked @($python, "-m", "ruff", "check", "src", "tests")
    }
}
