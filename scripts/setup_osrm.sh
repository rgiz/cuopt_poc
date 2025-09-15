#!/usr/bin/env bash
set -euo pipefail

echo "🗺️  Setting up OSRM for UK routing..."

# Configuration
OSRM_DATA_DIR="./data/osrm"
OSM_FILE="great-britain-latest.osm.pbf"
DOCKER_IMAGE="ghcr.io/project-osrm/osrm-backend:latest"

# Check if OSM file exists
if [ ! -f "$OSRM_DATA_DIR/$OSM_FILE" ]; then
    echo "❌ OSM file not found: $OSRM_DATA_DIR/$OSM_FILE"
    echo "Please run: wget http://download.geofabrik.de/europe/great-britain-latest.osm.pbf"
    echo "from the $OSRM_DATA_DIR directory"
    exit 1
fi

echo "📁 Working directory: $OSRM_DATA_DIR"
echo "📄 OSM file: $OSM_FILE"

# Step 1: Extract
echo "🔧 Step 1/3: Extracting OSM data (this will take several minutes)..."
docker run --rm -t -v "${PWD}/$OSRM_DATA_DIR:/data" $DOCKER_IMAGE \
    osrm-extract -p /opt/car.lua /data/$OSM_FILE

if [ $? -ne 0 ]; then
    echo "❌ osrm-extract failed"
    echo "💡 Try increasing Docker memory in Docker Desktop > Settings > Resources"
    exit 1
fi

echo "✅ Extract completed"

# Step 2: Partition
echo "🔧 Step 2/3: Partitioning graph (this will take a few minutes)..."
docker run --rm -t -v "${PWD}/$OSRM_DATA_DIR:/data" $DOCKER_IMAGE \
    osrm-partition /data/great-britain-latest.osrm

if [ $? -ne 0 ]; then
    echo "❌ osrm-partition failed"
    exit 1
fi

echo "✅ Partition completed"

# Step 3: Customize
echo "🔧 Step 3/3: Customizing (applying contraction hierarchies)..."
docker run --rm -t -v "${PWD}/$OSRM_DATA_DIR:/data" $DOCKER_IMAGE \
    osrm-customize /data/great-britain-latest.osrm

if [ $? -ne 0 ]; then
    echo "❌ osrm-customize failed"
    exit 1
fi

echo "✅ Customize completed"

# List generated files
echo "📋 Generated files:"
ls -la "$OSRM_DATA_DIR/"

echo ""
echo "🎉 OSRM preprocessing complete!"
echo "📝 You can now start the OSRM service with:"
echo "   docker-compose up osrm"
echo ""
echo "🧪 Test the service once running:"
echo "   curl 'http://localhost:5001/route/v1/driving/-0.1276,51.5074;-0.0899,51.5158?steps=false'"