# Calc_NTCP_CBCT

This repository contains Python code to calculate the **Normal Tissue Complication Probability (NTCP)** for CBCT (Cone Beam Computed Tomography) imaging doses in radiation therapy. The project focuses on evaluating the impact of CBCT imaging doses on normal tissues, specifically for breast cancer treatment scenarios (left and right breast), along with sensitivity analysis of NTCP parameters.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Files](#files)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data](#input-data)
- [Output Files](#output-files)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview
The code in this repository calculates NTCP for CBCT imaging doses used in radiation therapy, focusing on breast cancer treatment. It evaluates the impact of two imaging protocols (5 MU and 10 MU) on organs at risk (OARs) such as the contralateral breast, ipsilateral lung, contralateral lung, and heart. The NTCP calculations are performed using the **Logistic model** (for the breast) and the **Lyman-Kutcher-Burman (LKB) model** (for the lungs and heart). Additionally, the code performs statistical analysis (paired t-tests) and generates visualizations (boxplots) to compare the NTCP and mean doses between the two protocols.

## Features
- Calculate NTCP for CBCT imaging doses (5 MU and 10 MU protocols) for breast cancer treatment.
- Support for specific regions:
  - Left breast (`cbct_ntcp_breast_left.py`)
  - Right breast (`cbct_ntcp_breast_right.py`)
- Sensitivity analysis of NTCP parameters (`ntcp_sensitivity.py`).
- Statistical analysis using paired t-tests to compare NTCP and mean doses between 5 MU and 10 MU protocols.
- Visualization of results with boxplots for NTCP and mean doses.
- Modular code structure for easy integration into radiation therapy workflows.

## Files
- **`cbct_ntcp_breast_left.py`**: Calculates NTCP for CBCT imaging doses affecting the left breast, performs statistical analysis, and generates boxplots for visualization.
- **`cbct_ntcp_breast_right.py`**: Calculates NTCP for CBCT imaging doses affecting the right breast (similar functionality to the left breast script).
- **`ntcp_sensitivity.py`**: Performs sensitivity analysis on NTCP parameters to evaluate their impact on the results.
- **`README.md`**: Project documentation (this file).

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/thanhtaiphys/Calc_NTCP_CBCT.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Calc_NTCP_CBCT
   ```
3. Install the required dependencies (see [Dependencies](#dependencies) section).

## Usage
1. Ensure all dependencies are installed.
2. Prepare your input data in the correct directory structure (see [Input Data](#input-data) section).
3. Run the desired script. For example, to calculate NTCP for the left breast:
   ```bash
   python cbct_ntcp_breast_left.py
   ```
4. For the right breast:
   ```bash
   python cbct_ntcp_breast_right.py
   ```
5. For sensitivity analysis:
   ```bash
   python ntcp_sensitivity.py
   ```

## Input Data
The scripts expect DICOM files (RT Structure and RT Dose) in the following directory structure:
```
C:/RT_Project/data_cbct/Breast_CBCT/breast_left/
├── PatientID1/
│   ├── RS...dcm (RT Structure file)
│   ├── ...CBCT5...dcm (RT Dose file for 5 MU)
│   └── ...CBCT10...dcm (RT Dose file for 10 MU)
├── PatientID2/
│   ├── RS...dcm
│   ├── ...CBCT5...dcm
│   └── ...CBCT10...dcm
...
```
- **RT Structure files** should start with "RS".
- **RT Dose files** should contain "CBCT5" for 5 MU and "CBCT10" for 10 MU in their filenames.
- Update the `root_dir` variable in the script if your data is stored in a different location.

## Output Files
Running the scripts will generate the following output files:
- **`CBCT_Breast_Left_Summary_WithStats.csv`**: A CSV file containing NTCP and dose statistics (min, mean, max) for each patient, plan (5 MU or 10 MU), and ROI.
- **`CBCT_Breast_Left_Statistics.csv`**: A CSV file with statistical analysis results, including mean and standard deviation of doses and NTCP, along with p-values from paired t-tests.
- **`CBCT_NTCP_Boxplot_WithStats_for_Breast_Left.png`**: A boxplot comparing NTCP values between 5 MU and 10 MU protocols for each ROI.
- **`CBCT_Dose_Boxplot_WithStats_for_Breast_Left.png`**: A boxplot comparing mean doses between 5 MU and 10 MU protocols for each ROI.

(Note: Similar files will be generated for the right breast when running `cbct_ntcp_breast_right.py`.)

## Dependencies
The following Python libraries are required:
- Python 3.x
- NumPy (for numerical computations)
- Pandas (for data manipulation and CSV output)
- SciPy (for statistical tests and NTCP calculations)
- Matplotlib (for plotting boxplots)
- Seaborn (for enhanced visualization)
- dicompyler-core (for DICOM file parsing and DVH calculations)

Install them using pip:
```bash
pip install numpy pandas scipy matplotlib seaborn dicompyler-core
```

## Contributing
Contributions are welcome! If you have suggestions, improvements, or bug fixes:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a pull request.


