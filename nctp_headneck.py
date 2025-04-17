#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NTCP Analysis Program for Head & Neck after MV-CBCT

This program analyzes CBCT (Cone Beam Computed Tomography) data for head and neck cancer patients,
calculating and comparing NTCP (Normal Tissue Complication Probability) between
5MU and 10MU CBCT imaging protocols for various critical structures.

Key Features:
1. Processes DICOM data from radiation treatment plans
2. Calculates NTCP for multiple critical structures in head and neck region
3. Performs statistical comparison between 5MU and 10MU protocols
4. Generates visualization plots and statistical reports

Required Libraries:
- dicompylercore: DICOM data processing
- numpy: Numerical computations
- pandas: Tabular data processing
- scipy: Statistical analysis
- matplotlib: Basic plotting
- seaborn: Advanced visualization

Author: thanhtaiphys
Created: 2025-04-17 11:25:13 UTC
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, norm
import matplotlib.pyplot as plt
import seaborn as sns
from dicompylercore import dicomparser, dvhcalc

# Configuration constants
ROOT_DIR = r"C:/RT_Project/data_cbct/HeadNeck_CBCT"

# ROI definitions for head and neck structures
ROI_NAMES = [
    "OpticNerve_L", "OpticNerve_R",
    "Lens_L", "Lens_R",
    "Eye_L", "Eye_R",
    "Cochlea_L", "Cochlea_R",
    "Chiasm", "Brainstem", "SpinalCord+1mm",
    "Parotid_L", "Parotid_R"
]

# NTCP model parameters for each ROI
ROI_PARAMS = {
    "OpticNerve_L": {"model": "lkb", "d50": 55, "m": 0.15, "n": 0.2},
    "OpticNerve_R": {"model": "lkb", "d50": 55, "m": 0.15, "n": 0.2},
    "Lens_L": {"model": "lkb", "d50": 8,  "m": 0.1,  "n": 0.01},
    "Lens_R": {"model": "lkb", "d50": 8,  "m": 0.1,  "n": 0.01},
    "Eye_L": {"model": "lkb", "d50": 50, "m": 0.2,  "n": 0.3},
    "Eye_R": {"model": "lkb", "d50": 50, "m": 0.2,  "n": 0.3},
    "Cochlea_L": {"model": "lkb", "d50": 45, "m": 0.2,  "n": 0.3},
    "Cochlea_R": {"model": "lkb", "d50": 45, "m": 0.2,  "n": 0.3},
    "Chiasm": {"model": "lkb", "d50": 60, "m": 0.1,  "n": 0.2},
    "Brainstem": {"model": "lkb", "d50": 54, "m": 0.15, "n": 0.3},
    "SpinalCord+1mm": {"model": "lkb", "d50": 50, "m": 0.1,  "n": 0.1},
    "Parotid_L": {"model": "lkb", "d50": 39, "m": 0.2,  "n": 1},
    "Parotid_R": {"model": "lkb", "d50": 39, "m": 0.2,  "n": 1}
}

# Visualization settings
PLOT_CONFIG = {
    "figsize": (12, 7),
    "dpi": 300,
    "palette": {"5MU": "#2166AC", "10MU": "#B2182B"},
    "title_fontsize": 12,
    "rotation": 45,
    "title_pad": 20
}

def compute_ntcp_lkb(dose_bins: np.ndarray, counts: np.ndarray, 
                    d50: float, m: float, n: float) -> float:
    """
    Calculate NTCP using the Lyman-Kutcher-Burman (LKB) model.

    Args:
        dose_bins (np.ndarray): Array of dose values
        counts (np.ndarray): Array of volume counts for each dose bin
        d50 (float): Dose for 50% complication probability (Gy)
        m (float): Slope parameter
        n (float): Volume effect parameter

    Returns:
        float: NTCP value between 0 and 1
    """
    total_counts = np.sum(counts)
    if total_counts == 0:
        return 0.0
        
    v_fraction = np.array(counts) / total_counts
    doses = np.array(dose_bins[:len(v_fraction)])
    valid = v_fraction > 0
    
    if not np.any(valid):
        return 0.0
        
    deff = np.power(np.sum(v_fraction[valid] * (doses[valid] ** n)), 1/n)
    t = (deff - d50) / (m * d50)
    return float(norm.cdf(t))

def create_boxplot(data: pd.DataFrame, y_variable: str, title: str, 
                  output_filename: str) -> None:
    """
    Create and save a boxplot comparing 5MU and 10MU protocols.

    Args:
        data (pd.DataFrame): DataFrame containing the plot data
        y_variable (str): Column name for y-axis values
        title (str): Plot title
        output_filename (str): Name of output file to save the plot
    """
    plt.figure(figsize=PLOT_CONFIG["figsize"])
    sns.boxplot(data=data, x="ROI", y=y_variable, hue="Plan", 
                palette=PLOT_CONFIG["palette"])
    
    plt.title(title, fontsize=PLOT_CONFIG["title_fontsize"], 
              fontweight="bold", pad=PLOT_CONFIG["title_pad"])
    plt.xlabel("Structure")
    plt.ylabel(f"{y_variable}")
    plt.xticks(rotation=PLOT_CONFIG["rotation"])
    plt.legend(title="CBCT Protocol", loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()

def process_patient_data(patient_id: str, patient_path: str) -> list:
    """
    Process DICOM data and calculate NTCP for a single patient.

    Args:
        patient_id (str): Patient identifier
        patient_path (str): Path to patient's DICOM files

    Returns:
        list: List of dictionaries containing processed patient data
    """
    patient_records = []
    files = os.listdir(patient_path)
    
    try:
        struct_file = [f for f in files if f.startswith("RS")][0]
        dose_files = {
            "5MU": [f for f in files if "CBCT5" in f][0],
            "10MU": [f for f in files if "CBCT10" in f][0]
        }
    except IndexError:
        print(f"Missing required DICOM files for patient {patient_id}")
        return patient_records

    rtstruct = dicomparser.DicomParser(os.path.join(patient_path, struct_file))
    structure_dict = rtstruct.GetStructures()
    roi_name_to_id = {info['name']: rid for rid, info in structure_dict.items()}

    # Handle Brainstem naming variations
    for name in ["BrainStem+3mm", "Brainstem"]:
        if name in roi_name_to_id:
            roi_name_to_id["Brainstem"] = roi_name_to_id[name]

    for plan_type, dose_file in dose_files.items():
        rtdose = dicomparser.DicomParser(os.path.join(patient_path, dose_file))
        
        for roi_name in ROI_NAMES:
            if roi_name not in roi_name_to_id:
                print(f"⚠️ Missing ROI '{roi_name}' in {patient_id}")
                continue
                
            try:
                roi_id = roi_name_to_id[roi_name]
                dvh = dvhcalc.get_dvh(rtstruct.ds, rtdose.ds, roi_id)
                
                if dvh is None or len(dvh.bins) == 0:
                    continue
                    
                param = ROI_PARAMS[roi_name]
                ntcp = compute_ntcp_lkb(dvh.bins, dvh.counts, 
                                      param["d50"], param["m"], param["n"])
                
                patient_records.append({
                    "PatientID": patient_id,
                    "Plan": plan_type,
                    "ROI": roi_name,
                    "Min Dose": round(dvh.min, 2),
                    "Mean Dose": round(dvh.mean, 2),
                    "Max Dose": round(dvh.max, 2),
                    "NTCP": round(ntcp, 4)
                })
                
            except Exception as e:
                print(f"❌ Error processing {patient_id} - {roi_name}: {e}")
                
    return patient_records

def calculate_statistics(df: pd.DataFrame) -> list:
    """
    Calculate statistical comparisons between 5MU and 10MU protocols.

    Args:
        df (pd.DataFrame): DataFrame containing processed patient data

    Returns:
        list: List of dictionaries containing statistical analysis results
    """
    summary_stats = []
    for roi in df["ROI"].unique():
        g = df[df["ROI"] == roi]
        d5 = g[g["Plan"] == "5MU"].set_index("PatientID")
        d10 = g[g["Plan"] == "10MU"].set_index("PatientID")
        ids = d5.index.intersection(d10.index)
        
        if len(ids) < 2:
            continue
            
        d5_dose, d10_dose = d5.loc[ids]["Mean Dose"], d10.loc[ids]["Mean Dose"]
        d5_ntcp, d10_ntcp = d5.loc[ids]["NTCP"], d10.loc[ids]["NTCP"]
        dose_t, dose_p = ttest_rel(d5_dose, d10_dose)
        ntcp_t, ntcp_p = ttest_rel(d5_ntcp, d10_ntcp)
        
        summary_stats.append({
            "ROI": roi,
            "Dose_5MU_Mean": round(d5_dose.mean(), 2),
            "Dose_10MU_Mean": round(d10_dose.mean(), 2),
            "Dose_p-value": round(dose_p, 4),
            "NTCP_5MU_Mean": round(d5_ntcp.mean(), 4),
            "NTCP_10MU_Mean": round(d10_ntcp.mean(), 4),
            "NTCP_p-value": round(ntcp_p, 4)
        })
    
    return summary_stats

def main():
    """Main execution function for the NTCP analysis program."""
    # Process patient data
    records = []
    for patient_id in os.listdir(ROOT_DIR):
        pat_path = os.path.join(ROOT_DIR, patient_id)
        if not os.path.isdir(pat_path):
            continue
        records.extend(process_patient_data(patient_id, pat_path))

    # Create summary DataFrame and save results
    df = pd.DataFrame(records)
    df.to_csv("CBCT_HeadNeck_Summary.csv", index=False)
    print("\nSummary Statistics:")
    print(df.head())

    # Perform statistical analysis
    summary_stats = calculate_statistics(df)
    stats_df = pd.DataFrame(summary_stats)
    stats_df.to_csv("CBCT_HeadNeck_Statistics.csv", index=False)
    print("\nStatistical Analysis:")
    print(stats_df)

    # Create visualization plots
    create_boxplot(
        data=df,
        y_variable="NTCP",
        title="NTCP Comparison for Head & Neck Structures (5 MU vs 10 MU)",
        output_filename="CBCT_HeadNeck_NTCP_Boxplot.png"
    )

    create_boxplot(
        data=df,
        y_variable="Mean Dose",
        title="Mean Dose Comparison for Head & Neck Structures (5 MU vs 10 MU)",
        output_filename="CBCT_HeadNeck_Dose_Boxplot.png"
    )

    print(f"\n✅ Analysis completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

if __name__ == "__main__":
    main()