# Load the raw test data to Elasticsearch
# .\tools\load-es-test-data.ps1


Write-Host "Checking for required Python packages..." -ForegroundColor Cyan
try {
    $textacyOutput = & uv run python -c "import textacy; print('textacy version:', textacy.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "textacy package is installed" -ForegroundColor Green
    } else {
        throw "textacy import failed"
    }
}
catch {
    Write-Host "Error: The 'textacy' package is required but not installed." -ForegroundColor Red
    Write-Host ""
    Write-Host "To install it, run one of the following commands:" -ForegroundColor Yellow
    Write-Host "  uv sync --group es-data" -ForegroundColor Cyan
    Write-Host "  OR" -ForegroundColor Yellow
    Write-Host "  pip install textacy" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The textacy package is needed for the geonames_elasticsearch_loader.py script." -ForegroundColor Yellow
    exit 1
}



# Check if Elasticsearch is running
try {
    $response = Invoke-RestMethod -Uri "http://localhost:9200" -Method Get -TimeoutSec 5 -ErrorAction Stop
    Write-Host "Elasticsearch is already running." -ForegroundColor Green
}
catch {
    Write-Host "Elasticsearch is not running. Starting containers..." -ForegroundColor Yellow
    docker compose up -d
    
    # Wait for Elasticsearch to be ready
    Write-Host "Waiting for Elasticsearch to start..." -ForegroundColor Yellow
    do {
        Start-Sleep -Seconds 5
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:9200" -Method Get -TimeoutSec 5 -ErrorAction Stop
            $isReady = $true
        }
        catch {
            Write-Host "Still waiting..." -ForegroundColor Yellow
            $isReady = $false
        }
    } while (-not $isReady)
    
    Write-Host "Elasticsearch is ready!" -ForegroundColor Green
}

Write-Host "Creating mappings for the fields in the Geonames index..." -ForegroundColor Cyan
try {
    $mappingContent = Get-Content -Path "tools\data\geonames_mapping.json" -Raw
    $headers = @{
        'Content-Type' = 'application/json'
    }
    Invoke-RestMethod -Uri "http://localhost:9200/geonames" -Method Put -Body $mappingContent -Headers $headers
    Write-Host "Mappings created successfully." -ForegroundColor Green
}
catch {
    Write-Error "Failed to create mappings: $_"
}

Write-Host "Change disk availability limits..." -ForegroundColor Cyan
try {
    $settingsBody = @{
        transient = @{
            "cluster.routing.allocation.disk.watermark.low" = "10gb"
            "cluster.routing.allocation.disk.watermark.high" = "5gb"
            "cluster.routing.allocation.disk.watermark.flood_stage" = "4gb"
            "cluster.info.update.interval" = "1m"
        }
    } | ConvertTo-Json -Depth 3

    $headers = @{
        'Content-Type' = 'application/json'
    }
    Invoke-RestMethod -Uri "http://localhost:9200/_cluster/settings" -Method Put -Body $settingsBody -Headers $headers
    Write-Host "Disk limits updated successfully." -ForegroundColor Green
}
catch {
    Write-Error "Failed to update disk limits: $_"
}

Write-Host "`nLoading gazetteer into Elasticsearch..." -ForegroundColor Cyan
try {
    & uv run tools/geonames_elasticsearch_loader.py
    Write-Host "Gazetteer loaded successfully." -ForegroundColor Green
}
catch {
    Write-Error "Failed to load gazetteer: $_"
}

Write-Host "Done" -ForegroundColor Green
