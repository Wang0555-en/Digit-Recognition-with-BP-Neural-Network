@echo off
chcp 65001 >nul
cd /d "%~dp0"
"bin\x64\Release\BP.exe" --console
pause
