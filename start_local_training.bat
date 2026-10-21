@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=E:\Programes\ANACONDA\envs\pytorch311\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" -m my_scripts.train_yolov8 train --data E:/Dataset/dataset_four_classes/data.yaml %*
if errorlevel 1 pause
