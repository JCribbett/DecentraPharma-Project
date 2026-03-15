 @echo off
    setlocal enabledelayedexpansion

    REM Define base directory and autoresearch directory
    set "BASE_DIR=%~dp0"
    set "AUTORESEARCH_DIR=%~dp0autoresearch_module"

    REM --- Virtual Environment Activation ---
    set "PYTHON_EXE=python"
    if exist "%BASE_DIR%venv\Scripts\activate.bat" (
        echo Found virtual environment. Activating...
        call "%BASE_DIR%venv\Scripts\activate.bat"
        set "PYTHON_EXE=%BASE_DIR%venv\Scripts\python.exe"
        echo Python executable set to: !PYTHON_EXE!
    ) else (
        echo No virtual environment found. Using system python.
    )
    
    echo.
    echo Welcome to the DecentraPharma Research Control Script (v2)!
    echo Select an optimization task to start:
    echo.
    echo 1. Optimize HIV Model
    echo 2. Optimize Toxicity Model
    echo 3. Optimize Influenza Model
    echo 4. Optimize HIV and Toxicity Models (Concurrent)
    echo 5. Optimize All Models (HIV, Tox, Influenza) (Concurrent)
    echo 6. Exit
    echo.

    set /p choice="Enter your choice (1-6): "

    if "%choice%"=="1" (
        echo Starting HIV Model Optimization...
        start "HIV Optimization" !PYTHON_EXE! "%AUTORESEARCH_DIR%\optimizer.py" "train_hiv.py" "results_hiv.tsv"
    ) else if "%choice%"=="2" (
        echo Starting Toxicity Model Optimization...
        start "Toxicity Optimization" !PYTHON_EXE! "%AUTORESEARCH_DIR%\optimizer.py" "train_tox.py" "results_tox.tsv"
    ) else if "%choice%"=="3" (
        echo Starting Influenza Model Optimization...
        echo Note: Influenza optimizer not fully implemented in this version.
        rem start "Influenza Optimization" !PYTHON_EXE! "%AUTORESEARCH_DIR%\optimizer.py" "train_influenza.py" "results_influenza.tsv"
    ) else if "%choice%"=="4" (
        echo Starting HIV and Toxicity Model Optimizations (Concurrent)...
        start "HIV Optimization" !PYTHON_EXE! "%AUTORESEARCH_DIR%\optimizer.py" "train_hiv.py" "results_hiv.tsv"
        start "Toxicity Optimization" !PYTHON_EXE! "%AUTORESEARCH_DIR%\optimizer.py" "train_tox.py" "results_tox.tsv"
    ) else if "%choice%"=="5" (
        echo Starting All Model Optimizations (Concurrent)...
        start "HIV Optimization" !PYTHON_EXE! "%AUTORESEARCH_DIR%\optimizer.py" "train_hiv.py" "results_hiv.tsv"
        start "Toxicity Optimization" !PYTHON_EXE! "%AUTORESEARCH_DIR%\optimizer.py" "train_tox.py" "results_tox.tsv"
        echo Note: Influenza optimizer not fully implemented in this version.
    ) else if "%choice%"=="6" (
        echo Exiting.
    ) else (
        echo Invalid choice. Please enter a number between 1 and 6.
    )

    echo.
    echo Press any key to continue...
    pause > nul

    endlocal