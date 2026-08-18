<#
.SYNOPSIS
Wire a fresh git worktree of this repo up to the shared model weights.

.DESCRIPTION
`git worktree add` gives you only the *tracked* files -- about 2.5 MB. Everything
this project actually needs to run is gitignored: `models/` is ~11 GB and
`external/` holds clones of three out-of-tree repos. Re-downloading those per
worktree (setup_common.sh step 3 onward) would cost 11 GB and an hour each time,
so instead we point the new worktree at the one copy that already exists.

Two link strategies, because `models/` is *partially* tracked:

  - A model subdirectory with **no** tracked files (11 of 13 today) does not exist
    in a fresh worktree at all, so it can be a directory junction. Junctions are
    free, need no admin rights, and `git status` ignores them because the paths
    are gitignored anyway.
  - `models/sense-voice/` and `models/freyatts-ko-voiceA/` **do** carry tracked
    files (SenseVoice ships its own LICENSE, tokens.txt and test_wavs; FreyaTTS
    ships config.json). Those directories therefore already exist, and junctioning
    over them would either fail or hide the tracked files. For these we hardlink
    the individual untracked payload files instead. NTFS hardlinks cost no disk
    and no copy time, and every consumer opens these read-only -- llama.cpp and
    onnxruntime both mmap them -- so sharing one inode across worktrees is safe.

`data/` is deliberately NOT linked. It holds `conversations.db` (SQLite WAL) and
per-session audio, and `storage.py` resolves it off PROJECT_ROOT, so leaving it
per-worktree is what keeps two concurrent experiments from writing the same
database. Sharing it is the one link that would actually corrupt something.

.PARAMETER WorktreePath
The worktree to wire up. Defaults to the repo root this script lives in.

.PARAMETER SourceRepo
Where the already-populated models/ and external/ live. Defaults to the main
checkout, discovered via `git worktree list`.

.PARAMETER SkipVenv
Skip creating .venv-win. Useful when you only need the weights linked.

.EXAMPLE
  git worktree add -b exp/foo ..\flux-wt\foo main
  powershell -File ..\flux-wt\foo\scripts\_setup_worktree.ps1
#>
param(
    [string]$WorktreePath,
    [string]$SourceRepo,
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"

if (-not $WorktreePath) {
    $WorktreePath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$WorktreePath = (Resolve-Path $WorktreePath).Path

# The main checkout is the first line of `git worktree list` -- git always lists
# the primary working tree first, and it is the one that setup_common.sh populated.
if (-not $SourceRepo) {
    Push-Location $WorktreePath
    try {
        $first = (& git worktree list) | Select-Object -First 1
        # "<path> <sha> [<branch>]" -- path may contain spaces, so split on the sha.
        $SourceRepo = ($first -split '\s+[0-9a-f]{7,40}\s')[0].Trim()
    } finally { Pop-Location }
}
$SourceRepo = (Resolve-Path $SourceRepo).Path

if ($SourceRepo -eq $WorktreePath) {
    Write-Host "Source and target are the same checkout -- nothing to link." -ForegroundColor Yellow
    exit 0
}

Write-Host "source : $SourceRepo"
Write-Host "target : $WorktreePath"
Write-Host ""

# ---------------------------------------------------------------- models/
$srcModels = Join-Path $SourceRepo "models"
if (-not (Test-Path $srcModels)) {
    Write-Host "!! $srcModels does not exist -- run scripts/setup_windows.ps1 in the main checkout first." -ForegroundColor Red
    exit 1
}

# Ask git which files under models/ are tracked, so the two strategies stay in
# sync with reality instead of with a hardcoded list that rots.
Push-Location $SourceRepo
try { $tracked = @(& git ls-files "models/") } finally { Pop-Location }
$trackedDirs = @{}
foreach ($f in $tracked) {
    $parts = $f -split '/'
    if ($parts.Length -ge 2) { $trackedDirs[$parts[1]] = $true }
}

New-Item -ItemType Directory -Force -Path (Join-Path $WorktreePath "models") | Out-Null

$junctioned = 0; $hardlinked = 0; $skipped = 0
foreach ($dir in (Get-ChildItem $srcModels -Directory)) {
    $dst = Join-Path (Join-Path $WorktreePath "models") $dir.Name

    if (-not $trackedDirs.ContainsKey($dir.Name)) {
        # Fully untracked -> junction the whole directory.
        if (Test-Path $dst) { $skipped++; continue }
        New-Item -ItemType Junction -Path $dst -Target $dir.FullName | Out-Null
        Write-Host ("  junction  models/{0}" -f $dir.Name)
        $junctioned++
        continue
    }

    # Partially tracked -> hardlink each untracked payload file.
    $n = 0
    foreach ($f in (Get-ChildItem $dir.FullName -File -Recurse)) {
        $rel = $f.FullName.Substring($dir.FullName.Length).TrimStart('\')
        $relPosix = "models/" + $dir.Name + "/" + $rel.Replace('\','/')
        if ($tracked -contains $relPosix) { continue }   # git owns this one
        $target = Join-Path $dst $rel
        if (Test-Path $target) { continue }
        New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
        New-Item -ItemType HardLink -Path $target -Target $f.FullName | Out-Null
        $n++
    }
    if ($n -gt 0) {
        Write-Host ("  hardlink  models/{0}  ({1} file(s), tracked files left alone)" -f $dir.Name, $n)
        $hardlinked += $n
    } else { $skipped++ }
}

# -------------------------------------------------------------- external/
$srcExternal = Join-Path $SourceRepo "external"
$dstExternal = Join-Path $WorktreePath "external"
if ((Test-Path $srcExternal) -and -not (Test-Path $dstExternal)) {
    New-Item -ItemType Junction -Path $dstExternal -Target $srcExternal | Out-Null
    Write-Host "  junction  external/"
}

Write-Host ""
Write-Host ("models: {0} junctioned, {1} hardlinked, {2} already present" -f $junctioned, $hardlinked, $skipped)

# ------------------------------------------------------------------ venv
if (-not $SkipVenv) {
    if (Test-Path (Join-Path $WorktreePath ".venv-win")) {
        Write-Host ".venv-win already present -- skipping."
    } else {
        Write-Host ""
        Write-Host "Creating .venv-win (this takes a few minutes)..."
        Push-Location $WorktreePath
        try { & powershell -ExecutionPolicy Bypass -File (Join-Path $WorktreePath "scripts\setup_windows.ps1") }
        finally { Pop-Location }
    }
}

Write-Host ""
Write-Host "Done. Remember:" -ForegroundColor Green
Write-Host "  * set NOBODY_CPU_BUDGET per worktree so N concurrent runs don't each claim the whole machine"
Write-Host "  * serialize anything that reports milliseconds -- shared L3 and memory bandwidth are not partitioned by thread count"
Write-Host "  * never run two mic-using scripts at once; the audio device is a singleton"
