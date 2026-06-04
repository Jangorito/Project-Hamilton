Set-Location C:\Users\Jango\workspace\BeamNG

# Kill whatever is currently on port 5000
$listening = netstat -ano | Select-String '0.0.0.0:5000\s+.*LISTENING'
if ($listening) {
    $pid = ($listening.ToString().Trim() -split '\s+')[-1]
    taskkill /F /PID $pid
    Start-Sleep 2
}

git pull
.\venv\Scripts\python.exe scripts\launcher\app.py
