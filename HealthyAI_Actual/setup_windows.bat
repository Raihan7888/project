@echo off

setlocal

cd /d "%~dp0"

echo.
echo ==================================================
echo Healthy AI - Windows Setup
echo ==================================================
echo.

where py >nul 2>nul

if errorlevel 1 (
    echo Python launcher was not found.
    echo Install Python 3.14 and try again.
    pause
    exit /b 1
)


echo Creating Python 3.14 virtual environment...

py -3.14 -m venv .venv

if errorlevel 1 (
    echo.
    echo Could not create the virtual environment.
    pause
    exit /b 1
)


call .venv\Scripts\activate.bat


echo.
echo Upgrading pip...

python -m pip install --upgrade pip


echo.
echo Installing project dependencies...

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
)


echo.
echo Running project verification...

python verify_project.py


echo.
echo ==================================================
echo Setup completed.
echo ==================================================
echo.
echo Next:
echo.
echo 1. Configure Kaggle API credentials.
echo 2. Run:
echo    python download_dataset.py
echo.
echo 3. Run:
echo    python train_model.py
echo.
echo 4. Run:
echo    python quantize_model.py
echo.
echo 5. Run:
echo    python prepare_runtime.py
echo.
echo 6. Run:
echo    python app.py
echo.
pause