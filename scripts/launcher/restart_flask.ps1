Set-Location C:\Users\Jango\workspace\BeamNG

# Kill existing Flask process if running
Get-WmiObject Win32_Process -Filter "name='python.exe' and commandline like '%launcher%app.py%'" |
    ForEach-Object { $_.Terminate() }
Start-Sleep 1

git pull
.\venv\Scripts\python.exe scripts\launcher\app.py
