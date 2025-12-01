REM build exe file for Windows
pyinstaller --onefile --name prox-linux-reader --icon icon.ico prox-linux-reader.py
del prox-linux-reader.spec