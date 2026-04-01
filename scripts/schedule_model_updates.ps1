#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Automated Model Update Scheduler
    Runs periodic model updates based on configured schedules

.DESCRIPTION
    This script checks if model updates are due and triggers them automatically.
    Supports multiple update frequencies: micro (3 days), macro (7 days),
    major (30 days), and epoch (365 days).

.PARAMETER Force
    Force an update regardless of schedule

.PARAMETER UpdateType
    Force a specific update type (micro, macro, major, epoch)

.EXAMPLE
    .\schedule_model_updates.ps1
    .\schedule_model_updates.ps1 -Force
    .\schedule_model_updates.ps1 -UpdateType macro
#>

param(
    [switch]$Force,
    [ValidateSet("micro", "macro", "major", "epoch")]
    [string]$UpdateType
)

# Configuration
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$QuantTraderDir = Join-Path $ProjectRoot "quant_news_trader"
$ScriptsDir = Join-Path $ProjectRoot "scripts"
$LogsDir = Join-Path $QuantTraderDir "logs"
$ConfigDir = Join-Path $QuantTraderDir "configs"

# Ensure log directory exists
if (!(Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force
}

$LogFile = Join-Path $LogsDir "model_update_scheduler_$(Get-Date -Format 'yyyyMMdd').log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"
    Write-Host $LogEntry
    Add-Content -Path $LogFile -Value $LogEntry
}

function Get-LastUpdateTime {
    param([string]$UpdateType)

    $HistoryFile = Join-Path $LogsDir "model_update_history.jsonl"

    if (!(Test-Path $HistoryFile)) {
        return $null
    }

    # Read last line containing the update type
    $lastEntry = Get-Content $HistoryFile -Tail 1 | ConvertFrom-Json
    if ($lastEntry.update_type -eq $UpdateType) {
        return [DateTime]::Parse($lastEntry.timestamp)
    }

    # Search backwards for the update type
    $lines = Get-Content $HistoryFile
    for ($i = $lines.Length - 1; $i -ge 0; $i--) {
        try {
            $entry = $lines[$i] | ConvertFrom-Json
            if ($entry.update_type -eq $UpdateType) {
                return [DateTime]::Parse($entry.timestamp)
            }
        } catch {
            continue
        }
    }

    return $null
}

function Test-UpdateDue {
    param([string]$UpdateType, [int]$FrequencyDays)

    $lastUpdate = Get-LastUpdateTime -UpdateType $UpdateType

    if ($null -eq $lastUpdate) {
        Write-Log "No previous $UpdateType update found, update is due" "INFO"
        return $true
    }

    $daysSinceUpdate = (Get-Date) - $lastUpdate
    $isDue = $daysSinceUpdate.TotalDays -ge $FrequencyDays

    if ($isDue) {
        Write-Log "$UpdateType update due: $daysSinceUpdate days since last update" "INFO"
    } else {
        Write-Log "$UpdateType update not due: $daysSinceUpdate days since last update (needs $FrequencyDays days)" "DEBUG"
    }

    return $isDue
}

function Invoke-ModelUpdate {
    param([string]$UpdateType)

    Write-Log "Starting $UpdateType model update..." "INFO"

    try {
        # Activate virtual environment
        $VenvPath = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
        if (Test-Path $VenvPath) {
            & $VenvPath
        }

        # Change to quant trader directory
        Push-Location $QuantTraderDir

        # Run the update
        $pythonCmd = "python -m src.model_update --update-type $UpdateType"
        Write-Log "Executing: $pythonCmd" "DEBUG"

        $result = Invoke-Expression $pythonCmd 2>&1
        $exitCode = $LASTEXITCODE

        # Log output
        foreach ($line in $result) {
            Write-Log $line "INFO"
        }

        Pop-Location

        if ($exitCode -eq 0) {
            Write-Log "✅ $UpdateType update completed successfully" "SUCCESS"
            return $true
        } else {
            Write-Log "❌ $UpdateType update failed with exit code $exitCode" "ERROR"
            return $false
        }

    } catch {
        Write-Log "❌ $UpdateType update failed with exception: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Get-UpdateSchedule {
    # Read configuration to get update frequencies
    $ConfigFile = Join-Path $ConfigDir "config.yaml"

    if (!(Test-Path $ConfigFile)) {
        Write-Log "Configuration file not found: $ConfigFile" "ERROR"
        return $null
    }

    try {
        # Simple YAML parsing (basic implementation)
        $config = Get-Content $ConfigFile -Raw

        $schedule = @{
            "micro" = 3
            "macro" = 7
            "major" = 30
            "epoch" = 365
        }

        # Extract update frequencies from config (simplified)
        if ($config -match "micro_update_days:\s*(\d+)") {
            $schedule["micro"] = [int]$matches[1]
        }
        if ($config -match "macro_update_days:\s*(\d+)") {
            $schedule["macro"] = [int]$matches[1]
        }
        if ($config -match "major_update_days:\s*(\d+)") {
            $schedule["major"] = [int]$matches[1]
        }
        if ($config -match "epoch_update_days:\s*(\d+)") {
            $schedule["epoch"] = [int]$matches[1]
        }

        return $schedule

    } catch {
        Write-Log "Failed to parse configuration: $($_.Exception.Message)" "ERROR"
        return $null
    }
}

# Main execution
Write-Log "=== Model Update Scheduler Started ===" "INFO"

try {
    # Check if updates are enabled
    $ConfigFile = Join-Path $ConfigDir "config.yaml"
    if (!(Test-Path $ConfigFile)) {
        throw "Configuration file not found: $ConfigFile"
    }

    $configContent = Get-Content $ConfigFile -Raw
    if ($configContent -notmatch "enabled:\s*true") {
        Write-Log "Model updates are disabled in configuration" "INFO"
        exit 0
    }

    # Get update schedule
    $schedule = Get-UpdateSchedule
    if ($null -eq $schedule) {
        throw "Failed to load update schedule"
    }

    Write-Log "Update schedule loaded: micro=$($schedule['micro'])d, macro=$($schedule['macro'])d, major=$($schedule['major'])d, epoch=$($schedule['epoch'])d" "INFO"

    # Determine which updates to run
    $updatesToRun = @()

    if ($Force -or $UpdateType) {
        if ($UpdateType) {
            $updatesToRun = @($UpdateType)
            Write-Log "Forced update type: $UpdateType" "INFO"
        } else {
            $updatesToRun = @("micro", "macro", "major", "epoch")
            Write-Log "Forced all update types" "INFO"
        }
    } else {
        # Check which updates are due
        foreach ($updateType in @("micro", "macro", "major", "epoch")) {
            if (Test-UpdateDue -UpdateType $updateType -FrequencyDays $schedule[$updateType]) {
                $updatesToRun += $updateType
            }
        }
    }

    if ($updatesToRun.Count -eq 0) {
        Write-Log "No updates are currently due" "INFO"
        exit 0
    }

    Write-Log "Updates to run: $($updatesToRun -join ', ')" "INFO"

    # Run updates in order (micro -> macro -> major -> epoch)
    $updateOrder = @("micro", "macro", "major", "epoch")
    $successfulUpdates = 0

    foreach ($updateType in $updateOrder) {
        if ($updateType -in $updatesToRun) {
            Write-Log "--- Running $updateType update ---" "INFO"

            if (Invoke-ModelUpdate -UpdateType $updateType) {
                $successfulUpdates++
            }

            # Small delay between updates
            Start-Sleep -Seconds 5
        }
    }

    Write-Log "=== Scheduler completed: $successfulUpdates/$($updatesToRun.Count) updates successful ===" "INFO"

    if ($successfulUpdates -eq $updatesToRun.Count) {
        exit 0
    } else {
        exit 1
    }

} catch {
    Write-Log "Scheduler failed: $($_.Exception.Message)" "ERROR"
    exit 1
}