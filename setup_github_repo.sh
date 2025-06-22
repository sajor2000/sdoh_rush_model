#!/bin/bash
# GitHub Repository Setup Script
# ==============================
# 
# This script helps set up the SDOH Rush Model repository
# Repository: https://github.com/sajor2000/sdoh_rush_model

echo "🚀 Setting up SDOH Rush Model GitHub Repository"
echo "=" * 50

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the SDOH_Prediction_Model directory"
    exit 1
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
fi

# Add GitHub remote
echo "🔗 Adding GitHub remote..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/sajor2000/sdoh_rush_model.git
echo "✅ GitHub remote added"

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 .gitignore already exists"
else
    echo "✅ .gitignore configured"
fi

# Add all files
echo "📦 Adding files to Git..."
git add .
echo "✅ Files added"

# Check status
echo "📊 Repository status:"
git status --short

# Instructions for user
echo ""
echo "🎯 NEXT STEPS:"
echo "=============="
echo ""
echo "1. Review the files to be committed:"
echo "   git status"
echo ""
echo "2. Make your first commit:"
echo "   git commit -m \"Initial commit: SDOH prediction model with executive summary\""
echo ""
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. View your repository:"
echo "   https://github.com/sajor2000/sdoh_rush_model"
echo ""
echo "📋 KEY FILES FOR EXECUTIVES:"
echo "- executive_summary.html (linked images)"
echo "- executive_summary_embedded.html (self-contained)"
echo ""
echo "📊 ENHANCED VISUALIZATIONS:"
echo "- results/figures/ (12 publication-quality plots)"
echo "- All SHAP analysis and variable importance plots included"
echo ""
echo "🎉 Repository is ready for GitHub!"

# Optional: Show repository structure
echo ""
echo "📁 REPOSITORY STRUCTURE:"
echo "======================="
tree -L 2 -a 2>/dev/null || find . -type d -name ".*" -prune -o -type d -print | head -20