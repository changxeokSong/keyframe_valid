@echo off
rem ksl_validator 앱 원클릭 실행 스크립트 (Windows / run.sh의 Windows 버전).
rem conda 가상환경 또는 python venv 중 골라서 자동으로 만들고, 의존성 설치 후 실행한다.
rem
rem 사용법:
rem   run.bat [--conda^|--venv] ^<ksl_validator 명령 인자...^>
rem
rem 예:
rem   run.bat --conda fetch --origin-no 8240
rem   run.bat --venv validate --metadata sample.xlsx --limit 5
rem   run.bat gui

rem 콘솔 코드페이지를 UTF-8로 바꿔서 한글 출력이 깨지지 않게 함
chcp 65001 >nul

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "ENV_NAME=keyframe_valid"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "ENV_TYPE="

rem 콘솔 코드페이지를 바꿔도 파이썬 자체 stdout/stderr 인코딩이 안 맞으면
rem 한글 출력에서 에러가 날 수 있어 명시적으로 UTF-8을 강제한다
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if "%~1"=="-h" goto :usage
if "%~1"=="--help" goto :usage

if "%~1"=="--conda" (
    set "ENV_TYPE=conda"
    shift
) else if "%~1"=="--venv" (
    set "ENV_TYPE=venv"
    shift
)

if "%ENV_TYPE%"=="" (
    where conda >nul 2>nul
    if !errorlevel! equ 0 (
        set "ENV_TYPE=conda"
    ) else (
        set "ENV_TYPE=venv"
    )
)

rem batch의 %* 는 shift를 반영하지 않으므로, 남은 인자를 직접 다시 모은다
set "REMAINING="
:collect_args
if "%~1"=="" goto :args_done
set "REMAINING=!REMAINING! %1"
shift
goto :collect_args
:args_done

if "%ENV_TYPE%"=="conda" (
    where conda >nul 2>nul
    if !errorlevel! neq 0 (
        echo [run.bat] conda를 찾을 수 없습니다. run.bat --venv 를 사용하세요.
        exit /b 1
    )
    conda env list | findstr /b "%ENV_NAME%" >nul 2>nul
    if !errorlevel! neq 0 (
        echo [run.bat] conda 환경 '%ENV_NAME%' 생성 중...
        call conda create -y -n %ENV_NAME% python=3.11
    )
    call conda activate %ENV_NAME%
    if !errorlevel! neq 0 (
        echo [run.bat] conda activate 실패. Anaconda Prompt에서 실행 중인지 확인하세요.
        exit /b 1
    )
    set "PYTHON_BIN=python"
) else if "%ENV_TYPE%"=="venv" (
    if not exist "%VENV_DIR%" (
        echo [run.bat] python venv 생성 중... ^(%VENV_DIR%^)
        python -m venv "%VENV_DIR%"
    )
    call "%VENV_DIR%\Scripts\activate.bat"
    set "PYTHON_BIN=python"
) else (
    echo [run.bat] 알 수 없는 환경 타입: %ENV_TYPE%
    goto :usage
)

echo [run.bat] 의존성 확인/설치 중 ^(%ENV_TYPE% 환경^)...
%PYTHON_BIN% -m pip install -q -r requirements.txt

echo [run.bat] ksl_validator 실행 ^(%ENV_TYPE% 환경^): !REMAINING!
%PYTHON_BIN% -m ksl_validator !REMAINING!
exit /b %errorlevel%

:usage
echo 사용법: run.bat [--conda^|--venv] [ksl_validator 명령 인자...]
echo   --conda   conda 가상환경(keyframe_valid^) 사용
echo   --venv    python -m venv(.venv^) 사용
echo   옵션 생략 시 conda가 있으면 conda, 없으면 venv 자동 선택
echo.
echo 예:
echo   run.bat --conda fetch --origin-no 8240
echo   run.bat --venv validate --metadata sample.xlsx --limit 5
exit /b 1
