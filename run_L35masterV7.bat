@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >NUL
REM ==================== Configuración ====================
set "ROOT=C:\Python2025_SAC\GITHUBL58"
set "SRC=%ROOT%\src"
set "OUT=%ROOT%\out\L35"
set "SCRIPT=L35masterV7.py"
set "LOG=%OUT%\run_L35masterV7.log"

REM Opcional: forzar modo UTF-8 en Python
set "PYTHONUTF8=1"

if not exist "%OUT%" mkdir "%OUT%"

REM ==================== Selección de Python ====================
REM 1) Intenta usar el Python del venv del proyecto
set "PYEXE=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  REM 2) Si no existe, usa el Python del sistema
  set "PYEXE=python"
)

echo =============================================================== > "%LOG%"
echo L35 V7 - %DATE% %TIME% >> "%LOG%"
echo ROOT=%ROOT% >> "%LOG%"
echo PYEXE=%PYEXE% >> "%LOG%"
echo --------------------------------------------------------------- >> "%LOG%"

REM ==================== Ejecución ====================
pushd "%SRC%"
echo Ejecutando %SCRIPT% ...
"%PYEXE%" "%SRC%\%SCRIPT%" >> "%LOG%" 2>&1
set "ERR=%ERRORLEVEL%"
popd

REM ==================== Manejo de errores / salida ====================
if %ERR% NEQ 0 (
  echo [ERROR] Salida: %ERR%
  echo Revisa el log: "%LOG%"
  pause
  exit /b %ERR%
)

echo ✅ OK. Archivos en: "%OUT%"
if exist "%OUT%\index.html" start "" "%OUT%\index.html"
if exist "%OUT%\L35_profile_master.jpg" start "" "%OUT%\L35_profile_master.jpg"
if exist "%OUT%\L35_track_master.html" start "" "%OUT%\L35_track_master.html"

echo Log: "%LOG%"
pause
exit /b 0

