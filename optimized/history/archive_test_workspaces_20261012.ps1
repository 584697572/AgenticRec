$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$packageRoot = Join-Path $workspaceRoot 'optimized\AgenticRec'
$archiveRoot = Join-Path $workspaceRoot 'optimized\artifacts\archived_test_workspaces'
$workspacePrefix = $workspaceRoot.TrimEnd('\') + '\'
$moved = @()
foreach ($directory in (Get-ChildItem -LiteralPath $packageRoot -Force -Directory | Where-Object { $_.Name -like '.tmp_t19*' })) {
    $sourcePath = [IO.Path]::GetFullPath($directory.FullName)
    $targetPath = [IO.Path]::GetFullPath((Join-Path $archiveRoot $directory.Name))
    if (-not $sourcePath.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $targetPath.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected archive path' }
    if (Test-Path -LiteralPath $targetPath) { throw 'Refusing to overwrite historical workspace' }
    New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
    # Atomic same-volume directory rename avoids enumerating protected pytest
    # descendants; no ACL change and no recursive copy/delete fallback.
    [IO.Directory]::Move($sourcePath, $targetPath)
    $moved += $directory.Name
}
@{ status = 'ARCHIVED'; moved_count = $moved.Count; directories = $moved; deleted = 0 } | ConvertTo-Json
