@echo off
REM ============================================================
REM PySpark Project — Conda Environment Setup (Windows)
REM ============================================================

set ENV_NAME=pyspark-env
set PYTHON_VERSION=3.11

echo ==> Creating conda environment: %ENV_NAME% (Python %PYTHON_VERSION%)
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y

echo ==> Activating environment
call conda activate %ENV_NAME%

echo ==> Installing PySpark and OpenJDK via conda-forge
call conda install -c conda-forge pyspark openjdk=17 -y

echo ==> Installing additional Python dependencies
pip install pandas pyarrow numpy matplotlib jupyter pytest

echo ==> Pinning PYSPARK_PYTHON to avoid version mismatch
call conda env config vars set PYSPARK_PYTHON=python
call conda env config vars set PYSPARK_DRIVER_PYTHON=python

echo.
echo ============================================
echo   Setup complete!
echo   Activate with:  conda activate %ENV_NAME%
echo ============================================
pause