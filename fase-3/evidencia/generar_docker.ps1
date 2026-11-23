# Genera fase-3/evidencia/docker.txt con la evidencia de ejecucion de la Fase 3 en Docker.
# Ejecutar desde la RAIZ del repositorio:   .\fase-3\evidencia\generar_docker.ps1
# Requiere Docker Desktop en ejecucion y fase-2/data/housing.csv en su sitio.

$ErrorActionPreference = "Continue"
$salida = "fase-3\evidencia\docker.txt"
$imagen = "api-vivienda:3.0"
$nombre = "api-fase3"

if (-not (Test-Path "fase-3\api.py")) {
  Write-Host "ERROR: ejecute este script desde la raiz del repositorio (donde estan fase-2/ y fase-3/)." -ForegroundColor Red
  exit 1
}
if (-not (Test-Path "fase-2\data\housing.csv")) {
  Write-Host "ERROR: falta fase-2\data\housing.csv. Copielo con:" -ForegroundColor Red
  Write-Host '  Copy-Item fase-1\data\housing.csv fase-2\data\housing.csv'
  exit 1
}

New-Item -ItemType Directory -Force -Path "fase-3\evidencia" | Out-Null
Set-Content -Path $salida -Value "" -Encoding utf8

function Registrar([string]$texto) {
  $texto | Out-File -FilePath $salida -Encoding utf8 -Append
  Write-Host $texto
}
function Paso([string]$titulo, [scriptblock]$bloque) {
  Registrar ""
  Registrar "=============================================================="
  Registrar "PS> $titulo"
  Registrar "=============================================================="
  $r = (& $bloque 2>&1 | Out-String)
  Registrar $r.TrimEnd()
}

# El cliente de prueba solo usa la libreria estandar: sirve cualquier python.
$py = if (Test-Path "fase-3\venv\Scripts\python.exe") { "fase-3\venv\Scripts\python.exe" } else { "python" }

Registrar "Evidencia de ejecucion de la Fase 3 en Docker"
Registrar ("Fecha: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Registrar ("Equipo: " + $env:COMPUTERNAME + "  ·  Terminal: PowerShell " + $PSVersionTable.PSVersion)
Registrar ("Commit: " + (git rev-parse HEAD))

docker rm -f $nombre 2>&1 | Out-Null

Paso "docker --version" { docker --version }

Paso "docker build -f fase-3/Dockerfile -t $imagen --build-arg GIT_COMMIT=$(git rev-parse HEAD) ." {
  docker build -f fase-3/Dockerfile -t $imagen --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
}

Paso "docker run -d --rm --name $nombre -p 8000:8000 -v datos:ro -v modelos -e MODEL_VERSION=v1 $imagen" {
  docker run -d --rm --name $nombre -p 8000:8000 `
    -v "${PWD}/fase-2/data:/app/data:ro" `
    -v "${PWD}/fase-2/models:/app/models" `
    -e MODEL_VERSION=v1 `
    $imagen
}

Write-Host "Esperando a que la API responda..." -ForegroundColor Yellow
$listo = $false
foreach ($i in 1..40) {
  Start-Sleep -Seconds 2
  try { Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5 | Out-Null; $listo = $true; break } catch { }
}
if (-not $listo) {
  Registrar ""
  Registrar "ERROR: la API no respondio en 80 segundos. Registro del contenedor:"
  Registrar ((docker logs $nombre 2>&1 | Out-String).TrimEnd())
  docker stop $nombre 2>&1 | Out-Null
  exit 1
}

Paso "GET /health" {
  Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json -Depth 5
}

Paso "GET /versiones" {
  Invoke-RestMethod -Uri "http://localhost:8000/versiones" | ConvertTo-Json -Depth 5
}

Paso "python cliente_prueba.py   (prediccion con un registro real del proyecto)" {
  & $py fase-3\cliente_prueba.py --url http://localhost:8000
}

Paso "python cliente_prueba.py --quitar median_income   (entrada invalida: debe dar 422)" {
  & $py fase-3\cliente_prueba.py --url http://localhost:8000 --quitar median_income
}

Paso "POST /train   (entrena y guarda una version nueva)" {
  Invoke-RestMethod -Uri "http://localhost:8000/train" -Method Post `
    -ContentType "application/json" -Body "{}" -TimeoutSec 900 | ConvertTo-Json -Depth 5
}

Paso "GET /versiones   (la version nueva ya aparece)" {
  Invoke-RestMethod -Uri "http://localhost:8000/versiones" | ConvertTo-Json -Depth 5
}

Paso "dir fase-2\models   (en el equipo anfitrion: la version entrenada persiste fuera del contenedor)" {
  Get-ChildItem fase-2\models | Select-Object Mode, LastWriteTime, Name | Format-Table -AutoSize
}

Paso "docker stop $nombre" { docker stop $nombre }

Paso "docker run --rm $imagen python -m pytest   (las pruebas dentro de la imagen)" {
  docker run --rm $imagen python -m pytest
}

Registrar ""
Registrar "=============================================================="
Registrar "Fin de la evidencia."
Write-Host ""
Write-Host "Listo. Evidencia guardada en $salida" -ForegroundColor Green
