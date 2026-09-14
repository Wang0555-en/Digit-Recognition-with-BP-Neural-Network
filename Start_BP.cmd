@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "bin\x64\Release\BP.exe" (
    echo Please open BP.sln in Visual Studio and build Release / x64 first.
    pause
    exit /b 1
)
start "" "bin\x64\Release\BP.exe"
exit /b 0
