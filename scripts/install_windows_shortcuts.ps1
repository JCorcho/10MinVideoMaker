[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $projectRoot "Start 10MinVideoMaker.bat"
$icon = Join-Path $projectRoot "assets\10MinVideoMaker-icon.ico"
$commandPrompt = [Environment]::GetEnvironmentVariable("ComSpec")

foreach ($requiredPath in @($launcher, $icon, $commandPrompt)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required shortcut file was not found: $requiredPath"
    }
}

$desktop = [Environment]::GetFolderPath("Desktop")
$startMenuPrograms = Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"
$shortcutPaths = @(
    (Join-Path $desktop "10MinVideoMaker.lnk"),
    (Join-Path $startMenuPrograms "10MinVideoMaker.lnk")
)

$shell = New-Object -ComObject WScript.Shell
foreach ($shortcutPath in $shortcutPaths) {
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $commandPrompt
    $shortcut.Arguments = '/d /c ""{0}""' -f $launcher
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.IconLocation = "$icon,0"
    $shortcut.Description = "Start and configure the 10MinVideoMaker ComfyUI pipeline"
    $shortcut.WindowStyle = 1
    $shortcut.Save()
    Write-Output "Created shortcut: $shortcutPath"
}
