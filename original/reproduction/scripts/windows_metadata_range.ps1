param(
    [Parameter(Mandatory=$true)][string]$Url,
    [long]$RangeStart = 0,
    [long]$RangeEnd = -1,
    [int]$SuffixBytes = 0,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$taskAllowedUrls = @('https://files.grouplens.org/datasets/movielens/ml-10m.zip')
if ($Url -notin $taskAllowedUrls) { throw 'Unexpected metadata URL' }
if ($SuffixBytes -gt 0) {
    if ($SuffixBytes -gt 65557 -or $RangeEnd -ne -1) { throw 'Unexpected suffix request' }
} elseif ($RangeStart -lt 0 -or $RangeEnd -lt $RangeStart -or ($RangeEnd - $RangeStart + 1) -gt 2097152) { throw 'Unexpected metadata range request' }
$taskResolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$taskAllowedRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../logs/catalog_identity_20261005')) + [System.IO.Path]::DirectorySeparatorChar
if (-not $taskResolvedOutput.StartsWith($taskAllowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Output must remain in metadata audit log directory' }
if ([System.IO.File]::Exists($taskResolvedOutput)) { throw 'Existing range evidence must not be overwritten' }
$taskHttpRequest = [System.Net.HttpWebRequest]::Create($Url)
$taskHttpRequest.Timeout = 30000
$taskHttpRequest.ReadWriteTimeout = 30000
$taskHttpRequest.UserAgent = 'AgenticRec-metadata-audit'
if ($SuffixBytes -gt 0) { $taskHttpRequest.AddRange(-$SuffixBytes) } else { $taskHttpRequest.AddRange($RangeStart, $RangeEnd) }
$taskResponse = $taskHttpRequest.GetResponse()
try {
    $taskSize = if ($SuffixBytes -gt 0) { $SuffixBytes } else { $RangeEnd - $RangeStart + 1 }
    $taskLimit = if ([int]$taskResponse.StatusCode -eq 206) { $taskSize } else { 4096 }
    $taskBuffer = New-Object byte[] $taskLimit
    $taskStream = $taskResponse.GetResponseStream()
    $taskRead = 0
    while ($taskRead -lt $taskLimit) {
        $taskChunk = $taskStream.Read($taskBuffer, $taskRead, $taskLimit - $taskRead)
        if ($taskChunk -eq 0) { break }
        $taskRead += $taskChunk
    }
    $taskRecord = @{ status = [int]$taskResponse.StatusCode; content_range = $taskResponse.Headers['Content-Range']; bytes = $taskRead; url = $Url; tls_verification = 'Windows default certificate validation' }
    $taskFile = [System.IO.File]::Open($taskResolvedOutput, [System.IO.FileMode]::CreateNew)
    try { $taskFile.Write($taskBuffer, 0, $taskRead) } finally { $taskFile.Dispose() }
    $taskRecord | ConvertTo-Json -Compress
} finally { $taskResponse.Dispose() }
