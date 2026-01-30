# Project-wide installation and setup script
Write-Host "Installing C-RAG V3..." -ForegroundColor Green

# Check Python version
$pythonVersion = python --version 2>&1
if ($pythonVersion -notmatch "Python 3\.(10|11|12)") {
    Write-Host "ERROR: Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
    exit 1
}

Write-Host "Python version check passed" -ForegroundColor Green

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "`nDependencies installed successfully" -ForegroundColor Green

# Create necessary directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
$dirs = @("data", "checkpoints", "experiments", "logs")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Gray
    }
}

# Run tests
Write-Host "`nRunning tests..." -ForegroundColor Yellow
pytest tests/ -v --tb=short

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some tests failed" -ForegroundColor Yellow
} else {
    Write-Host "All tests passed" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nQuick start commands:" -ForegroundColor Yellow
Write-Host "  python -m crag.run_exp interactive    # Interactive mode"
Write-Host "  python -m uvicorn crag.api.server:app --reload  # Start API server"
Write-Host "  pytest tests/ -v                       # Run tests"
Write-Host ""
