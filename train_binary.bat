@echo off
cd /d "%~dp0"
set "FROZEN_PYTHON=E:\Programes\ANACONDA\envs\pytorch311\python.exe"
if not exist "%FROZEN_PYTHON%" set "FROZEN_PYTHON=python"
"%FROZEN_PYTHON%" -m my_scripts.train_frozen train --variant binary %*
exit /b %errorlevel%
