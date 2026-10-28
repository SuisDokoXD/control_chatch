$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot '.venv\Lib\site-packages'
    if ($args.Count -eq 0) {
        & py -3.13 (Join-Path $PSScriptRoot '程序\collect_dataset.py') --classes square
    } elseif ($args[0] -notlike '--*') {
        & py -3.13 (Join-Path $PSScriptRoot '程序\collect_dataset.py') --classes $args[0] @($args | Select-Object -Skip 1)
    } else {
        & py -3.13 (Join-Path $PSScriptRoot '程序\collect_dataset.py') @args
    }
    $captureExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
exit $captureExitCode
