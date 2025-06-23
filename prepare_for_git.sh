#!/bin/bash

# SDOH Model - Prepare for Git Release
# This script performs final cleanup before pushing to GitHub

echo "================================================"
echo "SDOH Risk Model - Preparing for Git Release"
echo "================================================"

# 1. Remove any .DS_Store files
echo "1. Removing .DS_Store files..."
find . -name ".DS_Store" -type f -delete

# 2. Remove any __pycache__ directories
echo "2. Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 3. Remove any .pyc files
find . -name "*.pyc" -type f -delete

# 4. Check for large files
echo "3. Checking for large files..."
echo "Files larger than 50MB:"
find . -type f -size +50M | grep -v ".git"

# 5. Check for sensitive patterns
echo "4. Checking for potential sensitive data..."
echo "Checking for API keys, passwords, tokens..."
grep -r -i -E "(api_key|apikey|password|passwd|token|secret)" --include="*.py" --include="*.json" --include="*.yaml" --include="*.yml" --exclude-dir=".git" --exclude-dir="sdoh_env" . | grep -v "# Check for" | head -20

# 6. List all CSV files (should be gitignored)
echo "5. CSV files found (these should be gitignored):"
find . -name "*.csv" -type f | grep -v ".git"

# 7. Verify .gitignore is working
echo "6. Files that Git is tracking (should not include data/env):"
git ls-files | grep -E "(\.csv|sdoh_env)" | head -10

# 8. Show repository statistics
echo "7. Repository statistics:"
echo "Total Python files: $(find . -name "*.py" -type f | wc -l)"
echo "Total Markdown files: $(find . -name "*.md" -type f | wc -l)"
echo "Total Notebooks: $(find . -name "*.ipynb" -type f | wc -l)"

echo ""
echo "================================================"
echo "FINAL CHECKLIST:"
echo "================================================"
echo "[ ] Review any large files listed above"
echo "[ ] Ensure no sensitive data was found"
echo "[ ] Verify all CSV files are gitignored"
echo "[ ] Update hardcoded paths to use config.py"
echo "[ ] Set your GitHub repository URL"
echo "[ ] Create GitHub repository if not exists"
echo ""
echo "If all checks pass, you can proceed with:"
echo "  git init (if needed)"
echo "  git add ."
echo "  git commit -m 'Initial release: SDOH Risk Model v2.0'"
echo "  git remote add origin https://github.com/[your-username]/sdoh-risk-model.git"
echo "  git push -u origin main"
echo ""
echo "================================================"