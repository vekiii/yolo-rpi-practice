# Aktiviranje C++ razvojnog okruženja

U novom PowerShell terminalu nalepi ceo blok:

```powershell
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath

if ($vsPath) {
    & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
    cl
} else {
    Write-Host "Nije pronadjena instalacija sa MSVC C++ alatima."
}
```

Očekuj naziv i verziju Microsoft C/C++ kompajlera za x64, pa `usage: cl ...`. To znači da je okruženje spremno. Aktivacija važi za trenutni terminal; ponovi je u novom terminalu.
