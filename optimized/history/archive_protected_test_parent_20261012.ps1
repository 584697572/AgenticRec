# Preserve four inaccessible old pytest roots without changing their ACLs.
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$workspacePrefix = $workspaceRoot.TrimEnd('\') + '\'
$packageRoot = Join-Path $workspaceRoot 'optimized\AgenticRec'
$cleanRoot = Join-Path $workspaceRoot 'optimized\.consolidation-package-tmp'
$archiveParent = Join-Path $workspaceRoot 'optimized\artifacts\archived_test_workspaces\protected_workspaces'
function Check-Path([string]$path) {
    $resolved = [IO.Path]::GetFullPath($path)
    if (-not $resolved.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Outside workspace' }
    if ((Test-Path -LiteralPath $resolved) -and ((Get-Item -LiteralPath $resolved -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected reparse point' }
    return $resolved
}
$packageRoot = Check-Path $packageRoot
$cleanRoot = Check-Path $cleanRoot
$archiveParent = Check-Path $archiveParent
if ((Test-Path -LiteralPath $cleanRoot) -or (Test-Path -LiteralPath $archiveParent)) { throw 'Refusing to overwrite' }
$remaining = @(Get-ChildItem -LiteralPath $packageRoot -Force | Where-Object { $_.Name -like '.tmp_t19*' })
if ($remaining.Count -ne 4) { throw 'Expected exactly four remaining old test roots' }
$items = @(Get-ChildItem -LiteralPath $packageRoot -Force | Where-Object { $_.Name -notlike '.tmp_t19*' })
New-Item -ItemType Directory -Path $cleanRoot | Out-Null
$moved = @()
$parentArchived = $false
try {
    foreach ($item in $items) {
        $sourcePath = Check-Path $item.FullName
        $targetPath = Check-Path (Join-Path $cleanRoot $item.Name)
        if ($item.PSIsContainer) { [IO.Directory]::Move($sourcePath, $targetPath) }
        else { [IO.File]::Move($sourcePath, $targetPath) }
        $moved += $item.Name
    }
    # The parent is owned by the project user. Rename it atomically, leaving
    # the restrictive children untouched rather than enumerating or deleting.
    [IO.Directory]::Move($packageRoot, $archiveParent)
    $parentArchived = $true
    [IO.Directory]::Move($cleanRoot, $packageRoot)
} catch {
    if ($parentArchived -and -not (Test-Path -LiteralPath $packageRoot)) {
        [IO.Directory]::Move($archiveParent, $packageRoot)
    }
    foreach ($name in $moved) {
        $from = Join-Path $cleanRoot $name
        $to = Join-Path $packageRoot $name
        if (Test-Path -LiteralPath $from) {
            if ([IO.Directory]::Exists($from)) { [IO.Directory]::Move($from, $to) }
            else { [IO.File]::Move($from, $to) }
        }
    }
    throw
}
@{ status = 'ARCHIVED'; protected_directories = 4; deleted = 0; acl_changes = 0 } | ConvertTo-Json
