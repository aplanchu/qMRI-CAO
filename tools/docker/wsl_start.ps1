$Command = "wsl --exec docker-compose up --build"

Write-Host "Starting mlflow server"

Push-Location (Split-Path $MyInvocation.MyCommand.Path)
Invoke-Expression $Command
Pop-Location