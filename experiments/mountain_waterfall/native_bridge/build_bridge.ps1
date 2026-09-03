param(
    [string]$IsaacRoot = "Y:\isaacsim",
    [string]$PhysXSource = "Y:\tools\PhysX-5.9.0-src"
)

$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Build = Join-Path $Here "build"
$Bin = Join-Path $Here "bin"
New-Item -ItemType Directory -Force -Path $Build, $Bin | Out-Null

$VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$Install = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $Install) { throw "MSVC Build Tools were not found" }
$VcVars = Join-Path $Install "VC\Auxiliary\Build\vcvars64.bat"

$Source = Join-Path $Here "physx_diffuse_bridge.cpp"
$Output = Join-Path $Bin "_physx_diffuse_bridge.pyd"
$Object = Join-Path $Build "physx_diffuse_bridge.obj"
$PythonInclude = Join-Path $IsaacRoot "kit\python\include"
$PythonLib = Join-Path $IsaacRoot "kit\python\libs\python312.lib"
$KitInclude = Join-Path $IsaacRoot "kit\dev\include"
$PhysXInclude = Join-Path $PhysXSource "physx\include"
$FoundationInclude = Join-Path $PhysXSource "physx\source\foundation\include"

$Command = @(
    'call "' + $VcVars + '" >nul &&',
    'cl /nologo /std:c++17 /EHsc /MD /O2 /LD /DNDEBUG /utf-8 /permissive- /Zc:__cplusplus /W4',
    '/I"' + $PythonInclude + '"',
    '/I"' + $KitInclude + '"',
    '/I"' + $PhysXInclude + '"',
    '/I"' + $FoundationInclude + '"',
    '/Fo"' + $Object + '"',
    '"' + $Source + '"',
    '/link /OUT:"' + $Output + '" "' + $PythonLib + '"'
) -join ' '

Write-Output $Command
& cmd.exe /d /s /c $Command
if ($LASTEXITCODE -ne 0) { throw "Native bridge compilation failed with exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $Output)) { throw "Compiler did not create $Output" }
Write-Output $Output
