@echo off
:: Auto Git Commit and Push Script (Only if changes exist)

:: Navigate to your project folder
cd /d "C:\Users\rishu narwal\Desktop\SVM_FDE"

:: Check for changes
for /f %%i in ('git status --porcelain') do set changes=true

if not defined changes (
    echo No changes detected. Exiting.
    pause
    exit /b
)

:: Stage changes
git add .

:: Commit with timestamp
set datetime=%date% %time%
git commit -m "Auto commit on %datetime%"

:: Push to main branch
git push origin main

echo Changes pushed successfully!
pause
