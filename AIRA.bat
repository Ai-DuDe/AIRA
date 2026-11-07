@echo off
rem Double-click to open the AIRA launcher.
cd /d "%~dp0"
python launcher.py
if errorlevel 1 pause
