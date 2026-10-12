# One-time migration. Native PowerShell performs all moves; nothing is deleted.
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$inventory = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'layout_20261012.json') -Encoding UTF8 | ConvertFrom-Json
if ($inventory.before_commit -ne '4919c35303472fa92ea933a6c7583b07cc63a0ce') { throw 'Unexpected source inventory' }
function Move-WithinWorkspace([string]$from, [string]$to) {
    $sourcePath = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $from))
    $targetPath = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $to))
    $workspacePrefix = $workspaceRoot.TrimEnd('\') + '\'
    if (-not $sourcePath.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $targetPath.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Move outside workspace' }
    if (-not (Test-Path -LiteralPath $sourcePath)) { throw "Missing source: $from" }
    if (Test-Path -LiteralPath $targetPath) { throw "Refusing to overwrite: $to" }
    $targetParent = Split-Path -Parent $targetPath
    New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    Move-Item -LiteralPath $sourcePath -Destination $targetPath
}
Move-WithinWorkspace 'AgenticRec' 'optimized/AgenticRec'
Move-WithinWorkspace 'RecAI' 'original/RecAI'
Move-WithinWorkspace 'reproduction' 'original/reproduction'
Move-WithinWorkspace 'reports' 'optimized/reports'
Move-WithinWorkspace 'data' 'optimized/data'
Move-WithinWorkspace 'artifacts' 'optimized/artifacts'
foreach ($name in $inventory.baseline_report_names) {
    Move-WithinWorkspace "optimized/reports/$name" "original/reports/$name"
}
foreach ($name in @('upstream_movie', 'upstream_audit', 'unicrs', 'metadata_audit')) {
    Move-WithinWorkspace "optimized/data/raw/$name" "original/data/raw/$name"
}
Move-WithinWorkspace 'optimized/data/metadata_private' 'original/data/metadata_private'
Move-WithinWorkspace 'optimized/data/eval_private/upstream_redial_a1_hashseed42' 'original/data/eval_private/upstream_redial_a1_hashseed42'
foreach ($name in @('README.md', 'STATUS.md', 'DECISIONS.md', 'TASKS.yaml')) {
    Move-WithinWorkspace $name "optimized/history/workspace/$name"
}
$oldSpecification = Get-ChildItem -LiteralPath $workspaceRoot -File | Where-Object { $_.Name -like 'AgenticRec_*.md' }
if (@($oldSpecification).Count -ne 1) { throw 'Expected exactly one historical specification' }
Move-WithinWorkspace $oldSpecification.Name "optimized/history/workspace/$($oldSpecification.Name)"
Write-Output 'MIGRATED: canonical original/ and optimized/; no deletion, no download.'
