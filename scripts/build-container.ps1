[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BuildImage
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$tag = 'anomalib-runtime-build:local'
$container = 'anomalib-runtime-export'

docker version | Out-Null
docker build --platform windows/amd64 --build-arg "LIMR_BUILD_IMAGE=$BuildImage" `
    -f (Join-Path $root 'docker/windows/Dockerfile') -t $tag $root

docker rm -f $container 2>$null
docker create --name $container $tag | Out-Null
$output = Join-Path $root 'out/AnomaRT'
New-Item -ItemType Directory -Force -Path $output | Out-Null
docker cp "${container}:C:\src\out\dist\AnomaRT\." $output
docker rm $container | Out-Null
Write-Host "Package exported to $output"
