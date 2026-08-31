$ErrorActionPreference = "Stop"
$dataRoot = Join-Path $env:LOCALAPPDATA "HAVoiceIO"
$voiceExecutable = Join-Path $PSScriptRoot "..\..\.venv\Scripts\voice-io.exe"
& $voiceExecutable --config (Join-Path $dataRoot "commands.toml") --recordings (Join-Path $dataRoot "recordings") run --assets (Join-Path $dataRoot "assets")
