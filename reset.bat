@echo off
setlocal

:: ============================================================
:: Настройки — замени пути и имена файлов
:: ============================================================
set "FOLDER1=E:\Desktop\GoL_AI\GoL_AI\shared_models"
set "KEEP_FILE=agent_v1.pth"

set "FOLDER2=E:\Desktop\GoL_AI\GoL_AI\shared_data"
set "DELETE_FILE=dataset.h5"

set "FOLDER3=E:\Desktop\GoL_AI\GoL_AI\test1\random_merged"
set "COPY_FILE=dataset.h5"
:: ============================================================

:: 1. Удалить из FOLDER1 всё, кроме KEEP_FILE
echo Deleting agents...
for %%F in ("%FOLDER1%\*") do (
    if /I not "%%~nxF"=="%KEEP_FILE%" (
        del "%%F"
        echo   Deleted: %%~nxF
    )
)

:: 2. Удалить из FOLDER2 файл DELETE_FILE
echo Deleting dataset...
if exist "%FOLDER2%\%DELETE_FILE%" (
    del "%FOLDER2%\%DELETE_FILE%"
    echo   Deleted: %DELETE_FILE%
) else (
    echo   Not found: %DELETE_FILE%
)

:: 3. Скопировать из FOLDER3 в FOLDER2 файл COPY_FILE
echo Copying agent v1
if exist "%FOLDER3%\%COPY_FILE%" (
    copy "%FOLDER3%\%COPY_FILE%" "%FOLDER2%\%COPY_FILE%"
    echo   Copied: %COPY_FILE%
) else (
    echo   Not found: %COPY_FILE%
)

echo.
echo Done!
pause