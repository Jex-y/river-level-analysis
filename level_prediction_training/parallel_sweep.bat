@echo off

set n=%1
set command=%2

@REM activate venv
call .\venv\Scripts\activate

@REM Run the command n times in parallel
for /l %%i in (1,1,%n%) do (
    start cmd /k %command%
)