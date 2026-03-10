# audit-loop.ps1
#
# Runs iterative audits with a completely fresh Claude context each iteration.
# State is preserved between sessions in .claude/audit-tracker.json.
# Each fresh session reads the tracker, audits the next group, updates the tracker, and exits.
# Groups rotate in order; after all 6 are covered the cycle repeats from the lowest audit_count.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\audit-loop.ps1
#   powershell -ExecutionPolicy Bypass -File .\audit-loop.ps1 -MaxIterations 10
#   powershell -ExecutionPolicy Bypass -File .\audit-loop.ps1 -MaxIterations 4 -Model claude-opus-4-6

param(
    [int]$MaxIterations = 10,
    [string]$Model = "claude-sonnet-4-6"
)

$promptFile = ".claude\AUDIT-PROMPT.md"
$trackerFile = ".claude\audit-tracker.json"

if (-not (Test-Path $promptFile)) {
    Write-Error "Audit prompt not found: $promptFile"
    exit 1
}

if (-not (Test-Path $trackerFile)) {
    Write-Error "Audit tracker not found: $trackerFile"
    exit 1
}

Write-Host ""
Write-Host "Habla Audit Loop"
Write-Host "Model:          $Model"
Write-Host "Max iterations: $MaxIterations"
Write-Host ""

for ($iteration = 1; $iteration -le $MaxIterations; $iteration++) {

    # Read tracker to show which group is next
    # Sort by audit_count first, then by declared order to preserve intended sequence
    $tracker = Get-Content $trackerFile -Raw | ConvertFrom-Json
    $nextGroup = $tracker.groups.PSObject.Properties |
        Sort-Object { $_.Value.audit_count }, { $_.Value.order } |
        Select-Object -First 1
    $groupLabel = if ($nextGroup) { $nextGroup.Value.label } else { "unknown" }
    $totalDone  = $tracker.total_iterations

    Write-Host "--- Iteration $iteration / $MaxIterations  (total audits so far: $totalDone) ---"
    Write-Host "Group: $groupLabel"
    Write-Host "Starting fresh Claude session..."
    Write-Host ""

    # Fresh session — no --continue, context is clean each iteration
    $promptContent = Get-Content $promptFile -Raw
    claude -p $promptContent --model $Model

    Write-Host ""
    Write-Host "Iteration $iteration complete."
    Write-Host ""
}

# Final summary
$tracker = Get-Content $trackerFile -Raw | ConvertFrom-Json
Write-Host "================================"
Write-Host "Audit loop finished."
Write-Host "Total iterations run: $MaxIterations"
Write-Host "Total audits recorded: $($tracker.total_iterations)"
Write-Host ""
Write-Host "Audit counts per group:"
$tracker.groups.PSObject.Properties |
    Sort-Object { $_.Value.order } |
    ForEach-Object { Write-Host "  $($_.Value.order). $($_.Value.label): $($_.Value.audit_count) pass(es)" }
Write-Host ""
Write-Host "To continue: run again with any -MaxIterations and -Model."
Write-Host "Tracker preserves all progress between runs."
Write-Host "Review findings: .dev\docs\BUG_REGISTRY.md"
