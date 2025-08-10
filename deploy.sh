#!/bin/bash

# PCG Heart Sound Analyzer Deployment Script
# This script sets up and deploys the application

set -e

echo "🫀 PCG Heart Sound Analyzer - Deployment Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p src/{models,preprocessing,features}
mkdir -p frontend
mkdir -p tests

# Copy your existing models to the models directory
echo -e "${YELLOW}📋 Please ensure your trained models are placed in the 'models' directory:${NC}"
echo "   - models/artifact_detector_randomforest.pkl"
echo "   - models/murmur_healthy_gradient_boosting.joblib"
echo "   - models/s1s2_classifier.pkl (optional)"

# Check if models exist
MISSING_MODELS=""
if [ ! -f "models/artifact_detector_randomforest.pkl" ]; then
    MISSING_MODELS="${MISSING_MODELS}\n   - artifact_detector_randomforest.pkl"
fi
if [ ! -f "models/murmur_healthy_gradient_boosting.joblib" ]; then
    MISSING_MODELS="${MISSING_MODELS}\n   - murmur_healthy_gradient_boosting.joblib"
fi

if [ ! -z "$MISSING_MODELS" ]; then
    echo -e "${RED}❌ Missing required model files:${MISSING_MODELS}${NC}"
    echo -e "${YELLOW}Please copy your trained models to the models/ directory before continuing.${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build and start services
echo "🐳 Building Docker containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check if API is running
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API server is running at http://localhost:8000${NC}"
else
    echo -e "${RED}❌ API server failed to start${NC}"
    echo "Checking logs..."
    docker-compose logs pcg-api
    exit 1
fi

# Check if frontend is running
if curl -f http://localhost/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend is running at http://localhost${NC}"
else
    echo -e "${RED}❌ Frontend failed to start${NC}"
    echo "Checking logs..."
    docker-compose logs nginx
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Deployment successful!${NC}"
echo ""
echo "📋 Service URLs:"
echo "   🌐 Frontend: http://localhost"
echo "   🔧 API: http://localhost:8000"
echo "   📖 API Docs: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart"
echo "   Update: docker-compose pull && docker-compose up -d"
echo ""
echo -e "${YELLOW}💡 Make sure to place your trained models in the 'models' directory${NC}"
echo -e "${YELLOW}💡 Check the logs if you encounter any issues: docker-compose logs${NC}"