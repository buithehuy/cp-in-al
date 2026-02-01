@echo off
REM Simple Windows batch script to run multiple strategies
REM Usage: run_experiments.bat [data] [strategies...]
REM Example: run_experiments.bat cifar100 entropy combined

setlocal enabledelayedexpansion

if "%1"=="" (
    set DATA=cifar10
) else (
    set DATA=%1
)

shift

REM Collect strategies
set STRATEGIES=
:parse_strategies
if "%1"=="" goto run_experiments
set STRATEGIES=%STRATEGIES% %1
shift
goto parse_strategies

:run_experiments
REM Default to all if none specified
if "%STRATEGIES%"=="" (
    set STRATEGIES=random entropy least_confidence margin cp_size cp_v_shaped combined combined_v_shaped
)

echo Running experiments on %DATA% with strategies:%STRATEGIES%
echo.

for %%s in (%STRATEGIES%) do (
    echo ========================================
    echo Running: %%s on %DATA%
    echo ========================================
    python src/train.py data=%DATA% strategy=%%s
    echo.
)

echo.
echo All experiments completed!
echo To plot results: python plot_results.py outputs/

endlocal
