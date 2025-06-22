# Clinical Implementation Guide

## Overview

This guide provides healthcare professionals with the information needed to implement and use the SDOH prediction model in clinical practice.

## Model Purpose

The SDOH prediction model identifies patients who are likely to have 2 or more unmet social determinants of health needs. This allows healthcare teams to:

- Target limited screening resources efficiently
- Reduce screening burden by 75%
- Maintain high sensitivity (61%) for detecting patients with needs
- Improve positive predictive value by 2.5x compared to universal screening

## Clinical Workflow

### 1. Patient Risk Scoring

For each patient encounter:

1. **Extract Required Features**: Collect the necessary data points from the EHR:
   - Demographics (age, sex)
   - Insurance information (financial class)
   - Geographic data (address for SVI/ADI calculation)
   - Visit history and utilization patterns

2. **Calculate Risk Score**: Use the model to generate a risk score (0-1 scale)

3. **Apply Clinical Threshold**: Compare score to established thresholds

### 2. Threshold Selection

Choose the appropriate threshold based on your clinical context:

| Threshold Type | Value | Screens | PPV | Sensitivity | Use Case |
|----------------|-------|---------|-----|-------------|----------|
| **Standard** | 0.5644 | 25% | 16.2% | 61.1% | Routine screening programs |
| **High PPV** | 0.7726 | 5% | 28.7% | 21.7% | Limited resources, high confidence needed |
| **High Sensitivity** | 0.4392 | 40% | 12.6% | 75.9% | Safety-net populations, comprehensive screening |

### 3. Screening Process

For patients flagged by the model:

1. **Administer Full SDOH Screening**: Use your organization's comprehensive SDOH assessment tool
2. **Document Results**: Record both screening results and model predictions
3. **Provide Interventions**: Connect patients with identified needs to appropriate resources

### 4. Quality Monitoring

Track these metrics monthly:

- **Positive Predictive Value**: Target 15-20%
- **Screening Rate**: Should match expected rate for chosen threshold
- **Fairness Metrics**: Monitor selection rates across demographic groups

## Implementation Timeline

### Phase 1: Pilot (Months 1-2)
- Deploy in 1-2 clinics
- Train staff on workflow
- Monitor initial performance

### Phase 2: Validation (Months 3-4)
- Expand to 5-10 clinics
- Validate performance metrics
- Refine workflow based on feedback

### Phase 3: Full Deployment (Months 5-6)
- System-wide implementation
- Establish ongoing monitoring
- Create standard operating procedures

## Staff Training Requirements

### Clinical Staff
- Understanding of SDOH and model purpose (30 minutes)
- Interpretation of risk scores (15 minutes)
- Modified screening workflow (30 minutes)

### IT/Informatics Staff
- Model integration and deployment (2 hours)
- Monitoring dashboard setup (1 hour)
- Troubleshooting procedures (1 hour)

## Model Limitations

1. **Does not predict specific SDOH needs**: Only identifies presence of 2+ needs
2. **Requires EHR integration**: Manual data entry reduces efficiency
3. **Performance varies by subgroup**: Lower sensitivity for some racial groups
4. **Needs annual retraining**: Performance may drift over time

## Troubleshooting

### Low Positive Predictive Value
- Check if patient population has changed
- Verify data quality and completeness
- Consider adjusting threshold

### Screening Rate Too High/Low
- Verify threshold implementation
- Check for data quality issues
- Review model predictions for outliers

### Fairness Concerns
- Monitor demographic breakdowns monthly
- Investigate sudden changes in group-specific rates
- Consider group-specific thresholds if disparities emerge

## Regulatory Considerations

- Document model performance regularly
- Maintain audit logs of predictions
- Ensure compliance with institutional AI governance
- Regular bias monitoring as required by organizational policy

## Contact and Support

For technical issues:
- IT Help Desk: [contact information]
- Model developers: [contact information]

For clinical questions:
- SDOH program coordinator: [contact information]
- Medical informatics team: [contact information]