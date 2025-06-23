# SDOH Risk Screening Model - Build Automation
# Author: Juan C. Rojas, MD, MS

.PHONY: help install install-conda clean test train analyze docker-build docker-run

# Default target
help:
	@echo "SDOH Risk Screening Model - Available Commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  install        - Install dependencies with pip"
	@echo "  install-conda  - Install dependencies with conda"
	@echo "  clean         - Remove cache and temporary files"
	@echo ""
	@echo "Model Operations:"
	@echo "  train         - Train the final model"
	@echo "  analyze       - Run comprehensive SHAP/TRIPOD analysis"
	@echo "  test          - Validate model performance"
	@echo ""
	@echo "Docker Operations:"
	@echo "  docker-build  - Build Docker container"
	@echo "  docker-run    - Run analysis in container"
	@echo ""

# Install dependencies with pip
install:
	@echo "Installing SDOH model dependencies..."
	python -m venv sdoh_env
	./sdoh_env/bin/pip install --upgrade pip
	./sdoh_env/bin/pip install -r requirements.txt
	@echo "✅ Installation complete. Activate with: source sdoh_env/bin/activate"

# Install dependencies with conda
install-conda:
	@echo "Installing SDOH model with conda..."
	conda env create -f environment.yml
	@echo "✅ Conda environment created. Activate with: conda activate sdoh_prediction_model"

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	find . -name "*.tmp" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Train the final model
train:
	@echo "Training SDOH prediction model..."
	python scripts/training/final_train.py
	@echo "✅ Model training complete"

# Run comprehensive analysis
analyze:
	@echo "Running comprehensive SHAP and TRIPOD-AI analysis..."
	python scripts/analysis/comprehensive_shap_tripod_analysis.py
	@echo "✅ Analysis complete"

# Validate model performance
test:
	@echo "Validating model performance..."
	python scripts/evaluation/verify_test_metrics.py
	@echo "✅ Model validation complete"

# Build Docker container
docker-build:
	@echo "Building SDOH model Docker container..."
	docker build -t sdoh-model:latest .
	@echo "✅ Docker container built successfully"

# Run analysis in Docker container
docker-run:
	@echo "Running SHAP analysis in Docker container..."
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/results:/app/results sdoh-model:latest python scripts/analysis/comprehensive_shap_tripod_analysis.py
	@echo "✅ Docker analysis complete"

# Full reproducible pipeline
reproduce: clean install train analyze test
	@echo "🎯 Full reproducible pipeline completed successfully!"