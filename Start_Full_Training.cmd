@echo off
chcp 65001 >nul
cd /d "%~dp0"
python tools\train_full.py %*
set "TRAIN_EXIT=%ERRORLEVEL%"
pause
exit /b %TRAIN_EXIT%
