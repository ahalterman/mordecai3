echo "Downloading Geonames gazetteer NL data..."
wget https://download.geonames.org/export/dump/NL.zip 
wget https://download.geonames.org/export/dump/admin1CodesASCII.txt
wget https://download.geonames.org/export/dump/admin2Codes.txt

unzip NL.zip && rm readme.txt && rm NL.zip

echo "Starting Elasticsearch container..."
docker compose up -d

echo "Creating mappings for the fields in the Geonames index..."
curl -XPUT 'localhost:9200/geonames' -H 'Content-Type: application/json' -d @tools/geonames_mapping.json

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

echo "\nRemoving downloaded files..."
rm NL.txt admin1CodesASCII.txt admin2Codes.txt

echo "Done"