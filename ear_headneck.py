#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Secondary Cancer Risk Analysis for Head & Neck CBCT Data

This script calculates Excess Absolute Risk (EAR) for selected Organs at Risk (ROIs)
in the head and neck region using CBCT (Cone Beam Computed Tomography) data. 
The analysis includes age-stratified results and log-scale visualizations.

Model: Schneider's Secondary Cancer Risk Model
Key Metrics:
- RED (Risk Equivalent Dose)
- OED (Organ Equivalent Dose)
- μ (Age-time modifying factor)
- EAR (Excess Absolute Risk)

Author: Dr. Thanh Tai Duong
Created: 15/04/2025
Last Modified: 2025-04-17 11:33:49
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration constants
PLOT_CONFIG = {
    "figsize_regular": (10, 6),
    "figsize_large": (12, 6),
    "dpi": 300,
    "fontsize": {
        "title": 14,
        "label": 12
    }
}

# Schneider's model parameters
MODEL_PARAMS = {
    "alpha_prime": 0.085,
    "gamma_e": -0.3,
    "gamma_a": 1.0,
    "age_attained": 70
}

# Baseline EAR values for each ROI (cases per 10,000 PY)
EAR0_DICT = {
    "SpinalCord+1mm": 0.5,
    "Brainstem": 0.5,
    "Parotid_L": 0.7,
    "Parotid_R": 0.7,
    "OpticNerve_L": 0.6,
    "OpticNerve_R": 0.6
}

def load_and_merge_data(cbct_file: str, patient_info_file: str) -> pd.DataFrame:
    """
    Load and merge CBCT data with patient information.

    Args:
        cbct_file (str): Path to CBCT summary data
        patient_info_file (str): Path to patient information data

    Returns:
        pd.DataFrame: Merged dataset with patient information
    """
    cbct_df = pd.read_csv(cbct_file)
    patient_df = pd.read_csv(patient_info_file)
    return cbct_df.merge(patient_df, left_on='PatientID', 
                        right_on='Patient_ID', how='left')

def compute_scr_metrics(row: pd.Series) -> pd.Series:
    """
    Calculate secondary cancer risk metrics for a single data row.

    Args:
        row (pd.Series): Input data row containing ROI, dose, and age information

    Returns:
        pd.Series: Calculated metrics (RED, OED, μ, EAR)
    """
    roi = row['ROI']
    dose = row['Mean Dose']
    age_exposure = row['Age_at_Treatment']
    
    if roi not in EAR0_DICT or pd.isna(age_exposure) or pd.isna(dose):
        return pd.Series([np.nan, np.nan, np.nan, np.nan])
    
    # Calculate Risk Equivalent Dose (RED)
    red = dose * np.exp(-MODEL_PARAMS["alpha_prime"] * dose)
    
    # Calculate age-time modifying factor (μ)
    mu = np.exp(
        MODEL_PARAMS["gamma_e"] * (age_exposure - 30) + 
        MODEL_PARAMS["gamma_a"] * np.log(MODEL_PARAMS["age_attained"] / 70)
    )
    
    # Calculate Excess Absolute Risk (EAR)
    ear0 = EAR0_DICT[roi]
    ear = ear0 * red * mu
    
    return pd.Series([red, red, mu, ear])

def calculate_summary_statistics(data: pd.DataFrame, roi_list: list) -> pd.DataFrame:
    """
    Calculate extended summary statistics for EAR values.

    Args:
        data (pd.DataFrame): Input data containing EAR values
        roi_list (list): List of ROIs to include in analysis

    Returns:
        pd.DataFrame: Summary statistics including mean, median, IQR, etc.
    """
    filtered_df = data[data['ROI'].isin(roi_list)].dropna(subset=['EAR'])
    grouped = filtered_df.groupby(['ROI', 'Plan'])['EAR']
    
    summary_stats = grouped.agg(
        mean='mean',
        std='std',
        count='count',
        median='median',
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    ).reset_index()
    
    summary_stats['IQR'] = summary_stats['Q3'] - summary_stats['Q1']
    return summary_stats

def create_log_scale_boxplot(data: pd.DataFrame, roi_list: list, 
                           output_file: str) -> None:
    """
    Create and save a log-scale boxplot comparing EAR values.

    Args:
        data (pd.DataFrame): Input data containing EAR values
        roi_list (list): List of ROIs to include in the plot
        output_file (str): Output file path for saving the plot
    """
    filtered_df = data[data['ROI'].isin(roi_list)].dropna(subset=['EAR'])
    
    plt.figure(figsize=PLOT_CONFIG["figsize_regular"])
    sns.boxplot(data=filtered_df, x='ROI', y='EAR', hue='Plan')
    
    plt.yscale('log')
    plt.title('EAR Comparison (5MU vs 10MU) – Head & Neck (Log Scale)', 
              fontsize=PLOT_CONFIG["fontsize"]["title"])
    plt.ylabel('Excess Absolute Risk (EAR) [Log Scale]', 
              fontsize=PLOT_CONFIG["fontsize"]["label"])
    plt.xlabel('Organ at Risk (ROI)', 
              fontsize=PLOT_CONFIG["fontsize"]["label"])
    plt.legend(title='CBCT Plan')
    plt.grid(True, which="both", axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=PLOT_CONFIG["dpi"])
    plt.close()

def analyze_age_groups(data: pd.DataFrame, roi_list: list) -> tuple:
    """
    Perform age-stratified analysis of EAR values.

    Args:
        data (pd.DataFrame): Input data containing EAR values
        roi_list (list): List of ROIs to include in analysis

    Returns:
        tuple: (age_group_summary DataFrame, plot_figure)
    """
    def age_group(age):
        if age < 40:
            return "<40"
        elif 40 <= age <= 60:
            return "40–60"
        else:
            return ">60"
    
    data['Age_Group'] = data['Age_at_Treatment'].apply(age_group)
    
    # Calculate age group statistics
    age_group_summary = data[data['ROI'].isin(roi_list)] \
        .groupby(['Age_Group', 'ROI', 'Plan'])['EAR'] \
        .agg(['mean', 'std', 'median', 'count']).reset_index()
    
    # Create visualization
    plt.figure(figsize=PLOT_CONFIG["figsize_large"])
    sns.barplot(
        data=age_group_summary,
        x='ROI',
        y='mean',
        hue='Age_Group',
        ci=None
    )
    
    plt.title('Mean EAR by Age Group and ROI (Head & Neck)', 
              fontsize=PLOT_CONFIG["fontsize"]["title"])
    plt.ylabel('Mean EAR (cases/10,000 PY)', 
              fontsize=PLOT_CONFIG["fontsize"]["label"])
    plt.xlabel('Organ at Risk (ROI)', 
              fontsize=PLOT_CONFIG["fontsize"]["label"])
    plt.legend(title='Age Group')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    return age_group_summary, plt.gcf()

def main():
    """Main execution function for the secondary cancer risk analysis."""
    # Load and prepare data
    merged_df = load_and_merge_data(
        "CBCT_HeadNeck_Summary.csv",
        "Patient_Info.csv"
    )
    
    # Calculate SCR metrics
    merged_df[['RED', 'OED', 'mu', 'EAR']] = merged_df.apply(compute_scr_metrics, axis=1)
    roi_list = list(EAR0_DICT.keys())
    
    # Calculate and export summary statistics
    summary_stats = calculate_summary_statistics(merged_df, roi_list)
    summary_stats.to_csv("CBCT_EAR_Summary_HeadNeck_WithMedian_IQR.csv", index=False)
    
    # Create log-scale boxplot
    create_log_scale_boxplot(
        merged_df, 
        roi_list,
        "CBCT_EAR_Boxplot_HeadNeck_LogScale.png"
    )
    
    # Perform and export age-stratified analysis
    age_group_summary, age_plot = analyze_age_groups(merged_df, roi_list)
    age_group_summary.to_csv("CBCT_EAR_AgeGroup_Summary_HeadNeck.csv", index=False)
    age_plot.savefig("CBCT_EAR_Barplot_ByAgeGroup_HeadNeck.png", 
                     dpi=PLOT_CONFIG["dpi"])
    plt.close()
    
    # Export detailed patient-level data
    filtered_df = merged_df[merged_df['ROI'].isin(roi_list)]
    filtered_df[['PatientID', 'Age_at_Treatment', 'Plan', 'ROI', 'EAR']].to_csv(
        "CBCT_EAR_Detail_ByPatient_HeadNeck.csv", index=False
    )

if __name__ == "__main__":
    main()