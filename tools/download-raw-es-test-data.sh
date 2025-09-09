# Download and prepare raw GeoNames data
# Right now this is the NL dump, but maybe in the future it will be a filtered
# version of the full dump. 
echo "Downloading Geonames gazetteer NL data..."
wget https://download.geonames.org/export/dump/NL.zip 
wget https://download.geonames.org/export/dump/admin1CodesASCII.txt
wget https://download.geonames.org/export/dump/admin2Codes.txt

unzip NL.zip && rm readme.txt && rm NL.zip
mv NL.txt tools/data/NL.txt
mv admin1CodesASCII.txt tools/data/admin1CodesASCII.txt
mv admin2Codes.txt tools/data/admin2Codes.txt