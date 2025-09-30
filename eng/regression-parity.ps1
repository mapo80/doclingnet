[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('refresh', 'test', 'all')]
    [string]$Command = 'all',

    [string[]]$Case,

    [string]$PythonCli,

    [string]$Version,

    [string]$GoldenRoot,

    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $PythonCli) {
    $PythonCli = if ($env:DOCLING_PYTHON_CLI) { $env:DOCLING_PYTHON_CLI } else { 'docling' }
}

if (-not $Version) {
    $Version = if ($env:GOLDEN_VERSION) { $env:GOLDEN_VERSION } else { 'v0.12.0' }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

if (-not $GoldenRoot) {
    $GoldenRoot = if ($env:DOCLING_PARITY_GOLDEN_ROOT) {
        $env:DOCLING_PARITY_GOLDEN_ROOT
    }
    if (-not $GoldenRoot) {
        $GoldenRoot = Join-Path $repoRoot 'dataset/golden'
    }
}

if (-not $Case -or $Case.Count -eq 0) {
    $Case = @('DLN043A-001', 'DLN043A-002')
}

$versionRoot = Join-Path $GoldenRoot $Version
$artifactsDir = Join-Path $repoRoot '.artifacts'
$resultsDir = Join-Path $artifactsDir 'test-results'

New-Item -ItemType Directory -Path $artifactsDir, $resultsDir -Force | Out-Null

$cases = @{
    'DLN043A-001' = @{ Source = 'dataset/2305.03393v1-pg9-img.png'; Directory = '2305.03393v1-pg9' }
    'DLN043A-002' = @{ Source = 'dataset/amt_handbook_sample.pdf'; Directory = 'amt_handbook_sample' }
}

function Ensure-PythonCli {
    if (-not (Get-Command -Name $PythonCli -ErrorAction SilentlyContinue)) {
        throw "Docling CLI '$PythonCli' was not found on PATH. Use --PythonCli or set DOCLING_PYTHON_CLI."
    }
}

function Ensure-CustomPackages {
    $customDir = Join-Path $repoRoot 'packages/custom'
    $missing = @()
    foreach ($package in 'EasyOcrNet.1.0.0.nupkg', 'Docling.LayoutSdk.1.0.2.nupkg', 'TableFormerSdk.1.0.0.nupkg') {
        $path = Join-Path $customDir $package
        if (-not (Test-Path -LiteralPath $path)) {
            $missing += $package
        }
    }

    if ($missing.Count -gt 0) {
        throw "Install the custom NuGet packages before running the regression parity suite. Missing: $($missing -join ', ')"
    }
}

function Invoke-RefreshCase {
    param([string]$CaseId)

    $descriptor = $cases[$CaseId]
    if (-not $descriptor) {
        throw "Unknown catalog case '$CaseId'."
    }

    $sourceRelative = $descriptor.Source
    $directoryName = $descriptor.Directory

    $sourcePath = Join-Path $repoRoot $sourceRelative
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Source asset '$sourceRelative' for case '$CaseId' was not found."
    }

    $caseRoot = Join-Path $versionRoot $directoryName
    $sourceDir = Join-Path $caseRoot 'source'
    $pythonDir = Join-Path $caseRoot 'python-cli'
    $diffsDir = Join-Path $caseRoot 'diffs'

    New-Item -ItemType Directory -Path $caseRoot, $sourceDir, $diffsDir -Force | Out-Null

    $sourceName = Split-Path -Path $sourcePath -Leaf
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $sourceDir $sourceName) -Force

    $notesPath = Join-Path $sourceDir 'notes.md'
    if (-not (Test-Path -LiteralPath $notesPath)) {
        "# $CaseId source notes", '', "- TODO: document provenance details for '$sourceName'." | Set-Content -LiteralPath $notesPath -Encoding UTF8
    }

    if (Test-Path -LiteralPath $pythonDir) {
        Remove-Item -LiteralPath $pythonDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $pythonDir -Force | Out-Null

    $cliArgs = @(
        '--input', $sourcePath,
        '--output', $pythonDir,
        '--assets', 'assets',
        '--markdown', 'docling.md',
        '--manifest', 'manifest.json',
        '--telemetry', 'telemetry.json',
        '--keep-debug-overlays'
    )

    if ($env:DOCLING_PARITY_EXTRA_ARGS) {
        $cliArgs += $env:DOCLING_PARITY_EXTRA_ARGS -split ' '
    }

    Write-Host "[DLN-043d] Refreshing golden artefacts for $CaseId using '$PythonCli'."
    & $PythonCli @cliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Docling CLI exited with code $LASTEXITCODE."
    }
}

function Invoke-Refresh {
    Ensure-PythonCli
    New-Item -ItemType Directory -Path $versionRoot -Force | Out-Null
    foreach ($caseId in $Case) {
        Invoke-RefreshCase -CaseId $caseId
    }
}

function Invoke-Tests {
    Ensure-CustomPackages
    Write-Host '[DLN-043d] Restoring .NET solution.'
    dotnet restore (Join-Path $repoRoot 'DoclingNet.sln') | Out-Host
    Write-Host '[DLN-043d] Executing regression parity tests.'
    $env:DOCLING_PARITY_GOLDEN_ROOT = $versionRoot
    dotnet test (Join-Path $repoRoot 'DoclingNet.sln') --no-build --filter 'FullyQualifiedName~Docling.Tests.Tooling.RegressionParityTests' --logger 'trx;LogFileName=RegressionParity.trx' --results-directory $resultsDir | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet test failed with exit code $LASTEXITCODE."
    }
}

switch ($Command) {
    'refresh' { Invoke-Refresh }
    'test' { Invoke-Tests }
    'all' {
        Invoke-Refresh
        if (-not $SkipTests) {
            Invoke-Tests
        }
    }
    default { throw "Unsupported command '$Command'." }
}
