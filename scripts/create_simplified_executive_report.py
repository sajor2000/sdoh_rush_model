#!/usr/bin/env python3
"""
Create Simplified Executive HTML Report
Focus on clarity, simple explanations, and practical guidance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import base64
import joblib
import warnings

warnings.filterwarnings('ignore')

def img_to_base64(img_path):
    """Convert image to base64 for HTML embedding"""
    if Path(img_path).exists():
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None

def create_simple_visual_explanations():
    """Create simple visual aids to explain concepts"""
    
    # Create threshold comparison visual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Health System threshold
    ax1.text(0.5, 0.9, 'Whole Health System', fontsize=20, ha='center', fontweight='bold')
    ax1.text(0.5, 0.75, 'Use 5% Threshold', fontsize=18, ha='center', color='#2E86C1')
    ax1.text(0.5, 0.6, '✓ Screen 35% of all patients', fontsize=14, ha='center')
    ax1.text(0.5, 0.5, '✓ Find 72% of those with needs', fontsize=14, ha='center')
    ax1.text(0.5, 0.4, '✓ 1 in 7 screened have needs', fontsize=14, ha='center')
    ax1.text(0.5, 0.25, 'Best for: Mixed age populations', fontsize=12, ha='center', style='italic')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0.05, 0.15), 0.9, 0.8, fill=False, edgecolor='#2E86C1', linewidth=3))
    
    # Geriatric threshold
    ax2.text(0.5, 0.9, 'Geriatric Clinics (65+)', fontsize=20, ha='center', fontweight='bold')
    ax2.text(0.5, 0.75, 'Use 8.4% Threshold', fontsize=18, ha='center', color='#E74C3C')
    ax2.text(0.5, 0.6, '✓ Screen 8% of senior patients', fontsize=14, ha='center')
    ax2.text(0.5, 0.5, '✓ Find 73% of those with needs', fontsize=14, ha='center')
    ax2.text(0.5, 0.4, '✓ 1 in 5 screened have needs', fontsize=14, ha='center')
    ax2.text(0.5, 0.25, 'Best for: Patients 65 and older', fontsize=12, ha='center', style='italic')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0.05, 0.15), 0.9, 0.8, fill=False, edgecolor='#E74C3C', linewidth=3))
    
    plt.tight_layout()
    plt.savefig('results/figures/threshold_comparison_simple.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return 'results/figures/threshold_comparison_simple.png'

def create_simplified_html_report():
    """Create simplified executive report focused on clarity"""
    
    # Load model metadata
    with open('models/scientific_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create simple threshold comparison
    threshold_visual = create_simple_visual_explanations()
    
    # Collect key figures
    figures = {
        'threshold_simple': threshold_visual,
        'model_performance': 'results/figures/jama/figure1_model_performance.png',
        'feature_importance': 'results/figures/feature_importance_professional_labels.png',
        'fairness_dashboard': 'results/figures/comprehensive_fairness_dashboard.png',
        'senior_flowchart': 'results/figures/senior_clinic_implementation_flowchart.png',
    }
    
    # Convert to base64
    embedded_figures = {}
    for name, path in figures.items():
        b64 = img_to_base64(path)
        if b64:
            embedded_figures[name] = b64
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SDOH Screening Tool - Simple Guide</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.8;
            color: #2C3E50;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #F8F9FA;
            font-size: 16px;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2C3E50;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        h2 {{
            color: #3498DB;
            margin-top: 40px;
            font-size: 2em;
            border-bottom: 3px solid #3498DB;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #2C3E50;
            font-size: 1.5em;
            margin-top: 30px;
        }}
        .key-message {{
            background-color: #E8F8F5;
            border-left: 5px solid #27AE60;
            padding: 20px;
            margin: 30px 0;
            font-size: 1.1em;
        }}
        .warning-box {{
            background-color: #FEF9E7;
            border-left: 5px solid #F39C12;
            padding: 20px;
            margin: 30px 0;
        }}
        .simple-box {{
            background-color: #EBF5FB;
            border: 2px solid #3498DB;
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
            text-align: center;
        }}
        .big-number {{
            font-size: 3em;
            font-weight: bold;
            color: #3498DB;
            margin: 10px 0;
        }}
        .explanation {{
            font-size: 1.1em;
            color: #5D6D7E;
            margin: 10px 0;
        }}
        .image-section {{
            margin: 40px 0;
            text-align: center;
            background-color: #F8F9FA;
            padding: 30px;
            border-radius: 10px;
        }}
        .image-section img {{
            max-width: 100%;
            border: 2px solid #E5E7E9;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .simple-explanation {{
            background-color: #FADBD8;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.05em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            font-size: 1.05em;
        }}
        th {{
            background-color: #3498DB;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #E5E7E9;
        }}
        tr:hover {{
            background-color: #F8F9FA;
        }}
        .threshold-card {{
            background-color: white;
            border: 3px solid #3498DB;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .step-by-step {{
            background-color: #D5F4E6;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        .step-by-step ol {{
            font-size: 1.1em;
            line-height: 2;
        }}
        .center-text {{
            text-align: center;
        }}
        .highlight {{
            background-color: #FEF9E7;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SDOH Screening Tool - Simple Guide</h1>
        <p class="center-text" style="font-size: 1.2em; color: #7F8C8D;">
            A tool to help identify patients who need help with social needs<br>
            <strong>Version 2.0 | Easy-to-Understand Edition</strong>
        </p>
        
        <div class="key-message">
            <h3 style="margin-top: 0;">🎯 What This Tool Does</h3>
            <p>This computer program helps us find patients who might need help with:</p>
            <ul>
                <li>🍎 <strong>Food</strong> - Not having enough to eat</li>
                <li>🏠 <strong>Housing</strong> - Unsafe or unstable living situations</li>
                <li>🚗 <strong>Transportation</strong> - Can't get to doctor appointments</li>
                <li>💡 <strong>Utilities</strong> - Can't pay for heat or electricity</li>
                <li>🤝 <strong>Safety</strong> - Concerns about personal safety</li>
            </ul>
            <p><strong>If someone has 2 or more of these problems, we want to help them!</strong></p>
        </div>
        
        <h2>Why We Need This Tool</h2>
        
        <div class="simple-box">
            <div class="big-number">6.6%</div>
            <div class="explanation">Out of every 100 patients, about 7 have social needs</div>
        </div>
        
        <div class="warning-box">
            <h3 style="margin-top: 0;">⚠️ The Challenge</h3>
            <p><strong>We want to help everyone, but we don't have enough staff to screen all patients right now.</strong></p>
            <p>So this tool helps us find the patients who are <span class="highlight">most likely to need help</span>, 
            so we can start there while we work on getting more resources.</p>
        </div>
        
        <h2>Which Threshold Should You Use?</h2>
        
        <div class="image-section">
            {'<img src="data:image/png;base64,' + embedded_figures.get('threshold_simple', '') + '" alt="Threshold Comparison">' if embedded_figures.get('threshold_simple') else ''}
            <p class="explanation"><strong>Choose based on your clinic type!</strong></p>
        </div>
        
        <div class="threshold-card">
            <h3 style="color: #2E86C1;">🏥 For the Whole Health System</h3>
            <table>
                <tr>
                    <td><strong>Use This Number:</strong></td>
                    <td><span style="font-size: 1.5em; color: #2E86C1;"><strong>5%</strong></span></td>
                </tr>
                <tr>
                    <td><strong>What Happens:</strong></td>
                    <td>The computer will flag 35 out of every 100 patients for screening</td>
                </tr>
                <tr>
                    <td><strong>Success Rate:</strong></td>
                    <td>About 14 out of those 35 will actually have social needs (that's good!)</td>
                </tr>
                <tr>
                    <td><strong>Who This Helps:</strong></td>
                    <td>Works well for all ages - young adults, middle-aged, and seniors</td>
                </tr>
            </table>
        </div>
        
        <div class="threshold-card" style="border-color: #E74C3C;">
            <h3 style="color: #E74C3C;">👴 For Geriatric Clinics (Patients 65+)</h3>
            <table>
                <tr>
                    <td><strong>Use This Number:</strong></td>
                    <td><span style="font-size: 1.5em; color: #E74C3C;"><strong>8.4%</strong></span></td>
                </tr>
                <tr>
                    <td><strong>What Happens:</strong></td>
                    <td>The computer will flag 8 out of every 100 senior patients for screening</td>
                </tr>
                <tr>
                    <td><strong>Success Rate:</strong></td>
                    <td>About 20 out of every 100 screened will have needs (1 in 5 - very good!)</td>
                </tr>
                <tr>
                    <td><strong>Why Different:</strong></td>
                    <td>Seniors have different patterns - fewer have needs but they're often more serious</td>
                </tr>
            </table>
        </div>
        
        <h2>How Well Does It Work?</h2>
        
        <div class="image-section">
            <h3>Model Performance - What The Charts Mean</h3>
            {'<img src="data:image/png;base64,' + embedded_figures.get('model_performance', '') + '" alt="Model Performance">' if embedded_figures.get('model_performance') else ''}
            
            <div class="simple-explanation">
                <h4>📊 What These Charts Show:</h4>
                <ul>
                    <li><strong>Top Left (ROC Curve):</strong> The higher the blue line, the better. Our score of 0.765 is "Good"!</li>
                    <li><strong>Top Right (Precision-Recall):</strong> Shows we're much better than random guessing</li>
                    <li><strong>Bottom Left (Calibration):</strong> The dots follow the diagonal line - this means when we say "30% chance", we're right!</li>
                    <li><strong>Bottom Right (Score Distribution):</strong> Shows how we separate high-risk from low-risk patients</li>
                </ul>
            </div>
        </div>
        
        <h2>What Information Does The Tool Use?</h2>
        
        <div class="image-section">
            <h3>Most Important Factors</h3>
            {'<img src="data:image/png;base64,' + embedded_figures.get('feature_importance', '') + '" alt="Feature Importance">' if embedded_figures.get('feature_importance') else ''}
            
            <div class="simple-explanation">
                <h4>🔍 What This Chart Shows:</h4>
                <p>The tool looks at information about where patients live and their demographics. The longer the bar, the more important that factor is.</p>
                <ul>
                    <li><strong>Blue bars:</strong> Economic factors (poverty, unemployment)</li>
                    <li><strong>Purple bars:</strong> Household factors (single parents, elderly)</li>
                    <li><strong>Orange bars:</strong> Housing and transportation</li>
                    <li><strong>Red bars:</strong> Area deprivation measures</li>
                </ul>
                <p><strong>Key Point:</strong> The tool mostly uses neighborhood data, NOT personal medical information!</p>
            </div>
        </div>
        
        <h2>Is The Tool Fair To Everyone?</h2>
        
        <div class="key-message" style="background-color: #D5F4E6; border-color: #27AE60;">
            <h3 style="margin-top: 0;">✅ YES! The Tool Is Fair</h3>
            <p>We carefully tested to make sure the tool works equally well for:</p>
            <ul>
                <li>All age groups (young, middle-aged, elderly)</li>
                <li>Both men and women</li>
                <li>All racial and ethnic groups</li>
            </ul>
            <p><strong>No group is unfairly excluded or over-screened!</strong></p>
        </div>
        
        <div class="image-section">
            <h3>Fairness Across Different Groups</h3>
            {'<img src="data:image/png;base64,' + embedded_figures.get('fairness_dashboard', '') + '" alt="Fairness Analysis">' if embedded_figures.get('fairness_dashboard') else ''}
            
            <div class="simple-explanation">
                <h4>📊 What This Shows:</h4>
                <p>Each chart compares how well the tool works for different groups:</p>
                <ul>
                    <li><strong>Similar bar heights = FAIR</strong> ✅</li>
                    <li>The tool finds about 70% of people with needs in ALL groups</li>
                    <li>No group is left behind or unfairly targeted</li>
                </ul>
            </div>
        </div>
        
        <h2>How To Use This Tool - Step by Step</h2>
        
        <div class="step-by-step">
            <h3 style="margin-top: 0;">🚀 For Health System Staff</h3>
            <ol>
                <li><strong>The computer runs automatically</strong> - It checks all patients using their address and age</li>
                <li><strong>High-risk patients are flagged</strong> - You'll see an alert in the medical record</li>
                <li><strong>Screen flagged patients</strong> - Use your normal SDOH screening questions</li>
                <li><strong>Connect to resources</strong> - If they have needs, connect them to food banks, housing help, etc.</li>
                <li><strong>Document everything</strong> - Record what help was provided</li>
            </ol>
        </div>
        
        <div class="image-section">
            <h3>Workflow for Senior Clinics</h3>
            {'<img src="data:image/png;base64,' + embedded_figures.get('senior_flowchart', '') + '" alt="Senior Workflow">' if embedded_figures.get('senior_flowchart') else ''}
            <p class="explanation">Special workflow designed for patients 65 and older</p>
        </div>
        
        <h2>Important Things To Remember</h2>
        
        <div class="warning-box">
            <h3 style="margin-top: 0;">📌 Key Points</h3>
            <ul style="font-size: 1.1em; line-height: 2;">
                <li>This tool is <strong>NOT perfect</strong> - it's right about 70% of the time</li>
                <li>It's a <strong>helper tool</strong> - clinical judgment is still important</li>
                <li>Our goal is to eventually screen <strong>EVERYONE</strong> - this helps us start with limited resources</li>
                <li>The tool uses <strong>neighborhood data</strong>, not personal medical records</li>
                <li>It's been tested to be <strong>fair to all groups</strong></li>
            </ul>
        </div>
        
        <h2>Resources Needed</h2>
        
        <table>
            <tr>
                <th>What You Need</th>
                <th>Why It's Important</th>
            </tr>
            <tr>
                <td>Social Workers</td>
                <td>To help connect patients with resources</td>
            </tr>
            <tr>
                <td>Community Partnerships</td>
                <td>Food banks, housing agencies, transportation services</td>
            </tr>
            <tr>
                <td>Staff Training</td>
                <td>2-4 hours to learn how to use the tool and respond to alerts</td>
            </tr>
            <tr>
                <td>Computer System Updates</td>
                <td>IT team needs to add the tool to your medical records system</td>
            </tr>
        </table>
        
        <h2>Next Steps</h2>
        
        <div class="step-by-step" style="background-color: #FEF9E7;">
            <h3 style="margin-top: 0;">📋 Getting Started Checklist</h3>
            <ol>
                <li>Choose your threshold: <strong>5% for general</strong> or <strong>8.4% for geriatric</strong></li>
                <li>Train your staff on what to do when patients are flagged</li>
                <li>Set up partnerships with community resources</li>
                <li>Start with a pilot in 1-2 clinics</li>
                <li>Track your results and adjust as needed</li>
            </ol>
        </div>
        
        <div class="key-message" style="margin-top: 40px;">
            <h3 style="margin-top: 0;">🎯 Remember Our Goal</h3>
            <p style="font-size: 1.2em;">
                This tool helps us use our limited resources wisely to help the most vulnerable patients NOW, 
                while we work toward our goal of screening EVERYONE for social needs.
            </p>
            <p style="font-size: 1.1em;">
                <strong>Together, we can identify and address the social factors that affect our patients' health!</strong>
            </p>
        </div>
        
        <div style="text-align: center; margin-top: 50px; padding: 30px; background-color: #F8F9FA; border-radius: 10px;">
            <h3>Questions?</h3>
            <p style="font-size: 1.1em;">
                Contact the Population Health team for help implementing this tool<br>
                <strong>We're here to support you!</strong>
            </p>
            <hr style="margin: 20px 0;">
            <p style="color: #7F8C8D;">
                SDOH Screening Tool v2.0 - Simplified Guide<br>
                Last Updated: {pd.Timestamp.now().strftime('%B %Y')}
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save simplified HTML report
    output_path = Path('results/simplified_executive_report.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Simplified executive report saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return str(output_path)

def main():
    """Generate simplified report"""
    print("Creating Simplified Executive Report...")
    
    # Create output directory
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report_path = create_simplified_html_report()
    
    print("\n✅ Simplified report created successfully!")
    print("This version:")
    print("- Uses simple, clear language")
    print("- Explains all technical concepts")
    print("- Clearly shows different thresholds")
    print("- Removes unnecessary complexity")
    print("- Focuses on practical implementation")

if __name__ == "__main__":
    main()