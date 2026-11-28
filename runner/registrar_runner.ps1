# Registra esta máquina como runner autohospedado del repo, solo para la demostración.
# Requisitos: GitHub CLI con sesión (gh auth login), Docker Desktop y Git Bash.
# Uso (PowerShell, desde la carpeta runner del repo):  .\registrar_runner.ps1 -Repo manassesmauricio-cloud/pipeline-inferencia-monitoreo-urp-mcd-mlops-2602
param([Parameter(Mandatory = $true)][string]$Repo, [string]$Carpeta = "$env:USERPROFILE\actions-runner-s14")
$ErrorActionPreference = 'Stop'
$version = (gh api repos/actions/runner/releases/latest --jq .tag_name).TrimStart('v')
New-Item -ItemType Directory -Force $Carpeta | Out-Null
Set-Location $Carpeta
if (-not (Test-Path .\config.cmd)) {
  $zip = "actions-runner-win-x64-$version.zip"
  Invoke-WebRequest "https://github.com/actions/runner/releases/download/v$version/$zip" -OutFile $zip
  Expand-Archive $zip -DestinationPath . -Force
}
$token = gh api -X POST "repos/$Repo/actions/runners/registration-token" --jq .token
# Runner temporal: se enciende con run.cmd solo durante la demostración y se da de baja con quitar_runner.ps1.
# (No se usa --ephemeral porque el workflow tiene dos trabajos, CI y CD, y ese modo acepta uno solo.)
.\config.cmd --url "https://github.com/$Repo" --token $token --name "laptop-docente-s14" --labels "urp-mlops" --unattended --replace
Write-Output "Runner registrado. Para que empiece a escuchar trabajos:  .\run.cmd   (Ctrl+C para detenerlo)"
