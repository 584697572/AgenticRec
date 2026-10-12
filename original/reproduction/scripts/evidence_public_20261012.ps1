param([string]$Output = 'artifacts/runs/upstream_evidence_20261012/public', [switch]$Supplement, [switch]$Focused, [switch]$Discussions)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$targetDir = [IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
if (-not $targetDir.StartsWith(($projectRoot + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) { throw 'Output outside workspace' }
if (Test-Path -LiteralPath $targetDir) { throw 'Refuse to overwrite evidence' }
[IO.Directory]::CreateDirectory($targetDir) | Out-Null
Add-Type -AssemblyName System.Net.Http
$handler = New-Object System.Net.Http.HttpClientHandler
$handler.UseCookies = $false
$handler.UseDefaultCredentials = $false
$handler.AllowAutoRedirect = $false
$client = New-Object System.Net.Http.HttpClient($handler)
$client.Timeout = [TimeSpan]::FromSeconds(25)
$client.DefaultRequestHeaders.UserAgent.ParseAdd('AgenticRec-public-evidence-audit/1.0')
$ledger = New-Object 'System.Collections.Generic.List[object]'
$bodyBytes = 0
$byteCap = 4194304
$requestCap = 18
$utf8 = New-Object System.Text.UTF8Encoding($false)
if ($Supplement -or $Focused -or $Discussions) {
    $priorPath = 'artifacts/runs/upstream_evidence_20261012/public/ledger.json'
    if ($Focused) { $priorPath = 'artifacts/runs/upstream_evidence_20261012/public_supplement/ledger.json' }
    if ($Discussions) { $priorPath = 'artifacts/runs/upstream_evidence_20261012/public_focused/ledger.json' }
    $prior = Get-Content -Raw -LiteralPath (Join-Path $projectRoot $priorPath) | ConvertFrom-Json
    $bodyBytes = [int]$prior.body_bytes
    foreach ($r in $prior.requests) { $ledger.Add($r) }
}

function Save-Ledger {
    $record = [ordered]@{ audit_date = '2026-10-12'; body_bytes = $script:bodyBytes; byte_cap = $byteCap; requests = @($ledger.ToArray()); request_cap = $requestCap; retries = 0; tls_verification_disabled = $false; authenticated_requests = 0; redirects_followed = 0; paid_llm_requests = 0 }
    [IO.File]::WriteAllText((Join-Path $targetDir 'ledger.json'), ($record | ConvertTo-Json -Depth 12), $utf8)
}

function Get-Public([string]$name, [string]$url, [int]$limit) {
    if ($ledger.Count -ge $requestCap -or $script:bodyBytes + $limit -gt $byteCap) { throw 'Audit network budget exceeded' }
    $uri = [Uri]$url
    if ($uri.Scheme -ne 'https' -or $uri.Host -notin @('api.github.com', 'raw.githubusercontent.com', 'arxiv.org', 'files.grouplens.org')) { throw 'Unapproved public source' }
    if ($name -notmatch '^[a-z0-9_]+$') { throw 'Invalid evidence name' }
    $entry = [ordered]@{ name = $name; url = $url; started_at_utc = [DateTime]::UtcNow.ToString('o'); status = 'REQUESTED'; body_bytes = 0; complete = $false; file = $null; sha256 = $null; error = $null }
    $ledger.Add($entry)
    $response = $null
    try {
        $response = $client.GetAsync($url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        $entry.status = [int]$response.StatusCode
        $entry.headers = $response.Headers.ToString() + $response.Content.Headers.ToString()
        $stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $buffer = New-Object byte[] 8192
        $memory = New-Object IO.MemoryStream
        try {
            $eof = $false
            while ($memory.Length -lt $limit) {
                $want = [Math]::Min($buffer.Length, $limit - [int]$memory.Length)
                $count = $stream.ReadAsync($buffer, 0, $want).GetAwaiter().GetResult()
                if ($count -eq 0) { $eof = $true; break }
                $memory.Write($buffer, 0, $count)
                $script:bodyBytes += $count
                $entry.body_bytes += $count
            }
            $entry.complete = $eof -or ($response.Content.Headers.ContentLength -eq $memory.Length)
            $body = $memory.ToArray()
        } finally { $stream.Dispose(); $memory.Dispose() }
        $path = Join-Path $targetDir ($name + '.body')
        [IO.File]::WriteAllBytes($path, $body)
        $entry.file = $name + '.body'
        $entry.sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $entry.complete) { $entry.error = 'BODY_LIMIT_REACHED; no continuation or larger fetch' }
    } catch { $entry.error = $_.Exception.GetType().FullName + ': ' + $_.Exception.Message }
    finally { if ($null -ne $response) { $response.Dispose() }; Save-Ledger }
    Write-Output ([ordered]@{name=$entry.name;status=$entry.status;body_bytes=$entry.body_bytes;complete=$entry.complete;error=$entry.error} | ConvertTo-Json -Compress)
}

try {
    if ($Discussions) {
        Get-Public 'issue_75_comments' 'https://api.github.com/repos/microsoft/RecAI/issues/75/comments?per_page=100' 131072
        Get-Public 'issue_107_comments' 'https://api.github.com/repos/microsoft/RecAI/issues/107/comments?per_page=100' 131072
        Get-Public 'issue_106_comments' 'https://api.github.com/repos/microsoft/RecAI/issues/106/comments?per_page=100' 131072
        return
    }
    if ($Focused) {
        Get-Public 'interecagent_issues' 'https://api.github.com/search/issues?q=repo%3Amicrosoft%2FRecAI%20InteRecAgent%20is%3Aissue&per_page=20' 262144
        Get-Public 'mapping_issues' 'https://api.github.com/search/issues?q=repo%3Amicrosoft%2FRecAI%20mapping%20is%3Aissue&per_page=10' 131072
        Get-Public 'license_issues' 'https://api.github.com/search/issues?q=repo%3Amicrosoft%2FRecAI%20license%20is%3Aissue&per_page=10' 131072
        return
    }
    if ($Supplement) {
        # The original commit-detail response exceeded its bound due to patch content.
        # Keep that prefix and use the small commit-list endpoint, without increasing its limit.
        Get-Public 'latest_commit_brief' 'https://api.github.com/repos/microsoft/RecAI/commits?sha=main&per_page=1' 32768
        $brief = $ledger[$ledger.Count - 1]
        if ($brief.status -eq 200 -and $brief.complete) {
            $commits = Get-Content -Raw -LiteralPath (Join-Path $targetDir 'latest_commit_brief.body') | ConvertFrom-Json
            $headSha = @($commits)[0].sha
            if ($headSha -notmatch '^[a-f0-9]{40}$') { throw 'Invalid current upstream commit' }
            Get-Public 'current_tree' ('https://api.github.com/repos/microsoft/RecAI/git/trees/' + $headSha + '?recursive=1') 786432
            Get-Public 'current_readme' ('https://raw.githubusercontent.com/microsoft/RecAI/' + $headSha + '/InteRecAgent/README.md') 65536
            Get-Public 'current_license' ('https://raw.githubusercontent.com/microsoft/RecAI/' + $headSha + '/LICENSE.txt') 16384
        }
        Get-Public 'paper_v4_html' 'https://arxiv.org/html/2308.16505v4' 786432
        return
    }
    Get-Public 'latest_commit' 'https://api.github.com/repos/microsoft/RecAI/commits/main' 131072
    $headEntry = $ledger[0]
    if ($headEntry.status -eq 200 -and $headEntry.complete) {
        $commit = Get-Content -Raw -LiteralPath (Join-Path $targetDir 'latest_commit.body') | ConvertFrom-Json
        $headSha = $commit.sha
        if ($headSha -notmatch '^[a-f0-9]{40}$') { throw 'Invalid current upstream commit' }
        Get-Public 'current_tree' ('https://api.github.com/repos/microsoft/RecAI/git/trees/' + $headSha + '?recursive=1') 786432
        Get-Public 'current_readme' ('https://raw.githubusercontent.com/microsoft/RecAI/' + $headSha + '/InteRecAgent/README.md') 65536
        Get-Public 'current_license' ('https://raw.githubusercontent.com/microsoft/RecAI/' + $headSha + '/LICENSE.txt') 16384
    }
    Get-Public 'issue_110' 'https://api.github.com/repos/microsoft/RecAI/issues/110' 65536
    Get-Public 'issue_110_comments' 'https://api.github.com/repos/microsoft/RecAI/issues/110/comments?per_page=100' 262144
    Get-Public 'closed_issues_page1' 'https://api.github.com/repos/microsoft/RecAI/issues?state=closed&per_page=100' 524288
    Get-Public 'paper_abstract' 'https://arxiv.org/abs/2308.16505' 131072
    Get-Public 'paper_v3_html' 'https://arxiv.org/html/2308.16505v3' 786432
    Get-Public 'movielens10m_terms' 'https://files.grouplens.org/datasets/movielens/ml-10m-README.html' 65536
} finally { Save-Ledger; $client.Dispose(); $handler.Dispose() }
