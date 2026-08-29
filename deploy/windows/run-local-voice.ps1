$ErrorActionPreference = "Stop"
$dataRoot = Join-Path $env:LOCALAPPDATA "LocalVoice"
$voiceExecutable = Join-Path $PSScriptRoot "..\..\.venv\Scripts\local-voice.exe"
& $voiceExecutable --config (Join-Path $dataRoot "commands.toml") --recordings (Join-Path $dataRoot "recordings") run --assets (Join-Path $dataRoot "assets")
