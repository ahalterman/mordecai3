# Load the raw test data to Elasticsearch
# bash tools/load-es-test-data.sh

# Check if textacy package is installed
echo "Checking for required Python packages..."
if uv run python -c "import textacy" 2>/dev/null; then
  echo "✅ textacy package is installed"
else
  echo "❌ Error: The 'textacy' package is required but not installed."
  echo ""
  echo "To install it, run one of the following commands:"
  echo "  uv sync --group es-data"
  echo "  OR"
  echo "  pip install textacy"
  echo ""
  echo "The textacy package is needed for the geonames_elasticsearch_loader.py script."
  exit 1
fi

# Check if Elasticsearch is running
if ! curl -s localhost:9200 > /dev/null 2>&1; then
  echo "Elasticsearch is not running. Starting containers..."
  docker compose up -d
  # Wait for Elasticsearch to be ready
  echo "Waiting for Elasticsearch to start..."
  until curl -s localhost:9200 > /dev/null 2>&1; do
    echo "Still waiting..."
    sleep 5
  done
  echo "Elasticsearch is ready!"
else
  echo "Elasticsearch is already running."
fi

echo "Creating mappings for the fields in the Geonames index..."
curl -XPUT 'localhost:9200/geonames' -H 'Content-Type: application/json' -d @tools/data/geonames_mapping.json

echo "Change disk availability limits..."
curl -X PUT "localhost:9200/_cluster/settings" -H 'Content-Type: application/json' -d'
{
  "transient": {
    "cluster.routing.allocation.disk.watermark.low": "10gb",
    "cluster.routing.allocation.disk.watermark.high": "5gb",
    "cluster.routing.allocation.disk.watermark.flood_stage": "4gb",
    "cluster.info.update.interval": "1m"
  }
}
'

echo "\nLoading gazetteer into Elasticsearch..."
uv run tools/geonames_elasticsearch_loader.py

echo "Done"