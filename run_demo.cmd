@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m unittest -v test_pipeline || goto failed
"%PY%" make_demo.py --out samples\synthetic || goto failed
"%PY%" run.py image --packed samples\synthetic\packed.png --out outputs\synthetic || goto failed
echo Done. See outputs\synthetic\comparison.png.
popd
if not "%~1"=="--no-pause" pause
exit /b 0

:failed
popd
if not "%~1"=="--no-pause" pause
exit /b 1
