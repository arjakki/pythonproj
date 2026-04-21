@echo off
echo Starting MySQL 8.4...
start /B "MySQL" "C:\Program Files\MySQL\MySQL Server 8.4\bin\mysqld.exe" --datadir="C:\ProgramData\MySQL\MySQL Server 8.4\Data" --port=3306
timeout /t 3 /nobreak >nul
echo MySQL is running on localhost:3306
echo To stop: taskkill /IM mysqld.exe /F
