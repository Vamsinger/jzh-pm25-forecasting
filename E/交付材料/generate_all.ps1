$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$TrainingPython = "D:/Anaconda/envs/pytorch/python.exe"
$RuntimeRoot = "C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt/dependencies"
$PresentationSkill = "C:/Users/Admin/.codex/plugins/cache/openai-primary-runtime/presentations/26.1119.11507/skills/presentations"

$env:RUNTIME_PYTHON = Join-Path $RuntimeRoot "python/python.exe"
$env:RUNTIME_NODE_MODULES = Join-Path $RuntimeRoot "node/node_modules"
$env:PRESENTATION_SKILL_DIR = $PresentationSkill
$RuntimeNode = Join-Path $RuntimeRoot "node/bin/node.exe"

Push-Location $ProjectRoot
try {
    & $TrainingPython "E/交付材料/generate_report.py"
    if ($LASTEXITCODE -ne 0) { throw "实验报告生成失败" }

    & $RuntimeNode "E/交付材料/generate_presentation.mjs"
    if ($LASTEXITCODE -ne 0) { throw "答辩幻灯片生成失败" }
}
finally {
    Pop-Location
}

Write-Host "已生成实验报告.pdf和答辩幻灯片.pptx"
