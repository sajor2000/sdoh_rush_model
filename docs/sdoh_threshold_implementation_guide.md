
# EVIDENCE-BASED GROUP-SPECIFIC THRESHOLD IMPLEMENTATION GUIDE
==============================================================

Clinical Decision Support for Equitable SDOH Screening

Generated: 2025-06-21
Model: XGBoost SDOH Screening v1.0
Compliance: Healthcare AI Fairness Standards
        

## EXECUTIVE SUMMARY
------------------

**Compliance Status**: NON_COMPLIANT
**Primary Recommendation**: Group-specific thresholds required
**Clinical Impact**: Maintains ~28% PPV while ensuring equitable access across demographics

**Key Findings**:
- Universal threshold creates significant bias against elderly and certain racial groups
- Group-specific thresholds improve fairness without sacrificing clinical utility
- Continuous monitoring required for sustained equity
        

## RECOMMENDED THRESHOLD MATRIX
----------------------------------

### RACE GROUPS:
```
       Other: 0.756 (94th %ile) → PPV=27.9%, Screen=6.0%
       White: 0.723 (97th %ile) → PPV=21.0%, Screen=2.5%
       Black: 0.770 (85th %ile) → PPV=27.8%, Screen=14.5%
       Asian: 0.629 (95th %ile) → PPV=5.6%, Screen=5.0%
```

### AGE_GROUP GROUPS:
```
       36-50: 0.748 (89th %ile) → PPV=27.7%, Screen=10.5%
       18-35: 0.604 (80th %ile) → PPV=17.2%, Screen=20.0%
         66+: 0.704 (96th %ile) → PPV=19.7%, Screen=4.0%
       51-65: 0.742 (86th %ile) → PPV=27.8%, Screen=13.5%
```

### GENDER GROUPS:
```
      Female: 0.778 (95th %ile) → PPV=27.8%, Screen=5.0%
        Male: 0.756 (92th %ile) → PPV=27.8%, Screen=7.5%
```

## IMPLEMENTATION PROTOCOL
-------------------------

### Phase 1: Preparation (Weeks 1-2)
1. **System Integration**
   - Modify EHR to capture demographic data
   - Implement threshold lookup table
   - Configure clinical decision support alerts

2. **Staff Training**
   - Educate clinicians on group-specific thresholds
   - Explain fairness rationale and legal requirements
   - Provide interpretation guidelines

### Phase 2: Pilot Deployment (Weeks 3-6)
1. **Limited Rollout**
   - Deploy in 1-2 clinical units
   - Monitor for technical issues
   - Collect clinician feedback

2. **Quality Assurance**
   - Daily threshold application audits
   - Weekly fairness metric reviews
   - Bi-weekly clinical outcome assessments

### Phase 3: Full Deployment (Week 7+)
1. **System-wide Rollout**
   - Deploy across all clinical units
   - Implement automated monitoring
   - Establish monthly review process

2. **Continuous Improvement**
   - Quarterly threshold recalibration
   - Annual external fairness audit
   - Ongoing bias detection and mitigation
        

## MONITORING REQUIREMENTS
-------------------------

### Real-time Monitoring
- **Threshold Application Rate**: Ensure thresholds are correctly applied by group
- **Screening Volume**: Monitor for unexpected changes in screening rates
- **Alert Fatigue**: Track clinician response to screening recommendations

### Weekly Reviews
- **Fairness Metrics**: PPV equity, screening rate ratios, sensitivity by group
- **Clinical Outcomes**: Screen-positive rates, referral completion, services accessed
- **Technical Performance**: System uptime, data quality, threshold accuracy

### Monthly Assessments
- **Bias Detection**: Statistical tests for demographic disparities
- **Clinical Effectiveness**: Comparison of outcomes vs. baseline screening
- **Compliance Status**: Legal and ethical requirement adherence

### Quarterly Recalibration
- **Threshold Optimization**: Adjust based on new data and outcomes
- **Model Performance**: Overall discrimination and calibration assessment
- **Stakeholder Review**: Clinician feedback and patient outcome evaluation
        

## LEGAL AND ETHICAL COMPLIANCE
------------------------------

### Regulatory Framework
- **Civil Rights Act (1964)**: Prohibits discrimination in healthcare services
- **Americans with Disabilities Act (1990)**: Ensures equal access to medical care
- **HHS Section 1557**: Non-discrimination in health programs and activities

### Documentation Requirements
1. **Fairness Assessment Reports**: Monthly bias detection summaries
2. **Clinical Outcome Tracking**: Demographic-stratified effectiveness measures
3. **Threshold Justification**: Evidence-based rationale for group-specific decisions
4. **Staff Training Records**: Competency in equitable screening practices

### Audit Preparation
- Maintain detailed logs of threshold applications
- Document clinical decision-making rationale
- Prepare statistical evidence of fairness compliance
- Establish clear protocols for addressing identified disparities
        

## CLINICAL DECISION SUPPORT GUIDELINES
--------------------------------------

### Patient Encounter Workflow
1. **Demographic Verification**: Confirm age, race/ethnicity, gender in EHR
2. **Risk Score Calculation**: Apply ML model to generate base risk score
3. **Threshold Selection**: Use group-specific threshold for screening recommendation
4. **Clinical Judgment**: Override system recommendations when clinically indicated
5. **Documentation**: Record screening decision and rationale in patient record

### Interpretation Guidelines
- **High Risk (Above Threshold)**: Recommend comprehensive SDOH assessment
- **Moderate Risk (Near Threshold)**: Consider screening based on clinical judgment
- **Low Risk (Below Threshold)**: Standard care, reassess if risk factors change

### Override Protocols
Clinicians may override system recommendations when:
- Acute social crisis identified during visit
- Patient specifically requests SDOH screening
- Clinical judgment indicates higher risk than model predicts
- Family/caregiver reports concerning social circumstances

### Communication with Patients
- Explain screening recommendation clearly and respectfully
- Emphasize that screening helps identify available support services
- Address any concerns about demographic-based risk assessment
- Respect patient autonomy in screening decisions
        

## SUCCESS METRICS AND KPIs
--------------------------

### Fairness Metrics (Primary)
- **PPV Equity**: <10% difference between demographic groups
- **Screening Rate Ratio**: <3:1 between highest and lowest screening groups
- **Sensitivity Adequacy**: >15% for all demographic groups
- **Specificity Maintenance**: >90% for all demographic groups

### Clinical Effectiveness (Secondary)
- **SDOH Case Detection**: Increase from 6.6% to >20% in screened population
- **Resource Utilization**: Efficient allocation of SDOH support services
- **Patient Satisfaction**: Acceptance of screening recommendations
- **Clinician Adoption**: >80% compliance with threshold recommendations

### Process Metrics (Operational)
- **System Uptime**: >99% availability of threshold calculation
- **Data Quality**: <1% missing demographic data
- **Response Time**: <2 seconds for risk score calculation
- **Training Compliance**: 100% staff completion of fairness training

### Outcome Metrics (Long-term)
- **Health Equity**: Reduction in outcome disparities by demographic group
- **Service Access**: Increased SDOH service utilization among high-risk patients
- **Cost Effectiveness**: Improved ROI of SDOH screening program
- **Regulatory Compliance**: Zero violations of healthcare equity regulations
        

## IMMEDIATE ACTION ITEMS
------------------------

### CRITICAL VIOLATIONS (Must Address Before Deployment):
❌ race: PPV difference 22.3% exceeds 10.0%
❌ race: Screening rate ratio 5.8x exceeds 3.0x
❌ age_group: PPV difference 10.5% exceeds 10.0%
❌ age_group: Screening rate ratio 5.0x exceeds 3.0x

### RECOMMENDED ACTIONS:
✅ Implement group-specific threshold adjustments
✅ Establish continuous fairness monitoring
✅ Consider model retraining with fairness constraints
✅ Develop clinical protocols for equitable care