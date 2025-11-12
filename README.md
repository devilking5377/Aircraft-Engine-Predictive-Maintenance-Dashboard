# Aviation Component Anomaly Reporting Dashboard

A comprehensive interactive dashboard for predictive maintenance analysis of aircraft engines using sensor data and Remaining Useful Life (RUL) estimates. Built with Dash/Plotly and designed to explore the CMAPSS-style predictive maintenance dataset.

## Overview

This project provides a **CSV-backed interactive dashboard** using Dash and Plotly:

- **`main_csv.py`** — Standalone application for local CSV data
  - Loads data from `raw_sensor_data.csv` and `component_rul.csv`
  - Zero external dependencies (completely offline)
  - Fast, lightweight analysis for exploration and testing

## Dataset

The dashboard is designed to work with the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset:

- **`raw_sensor_data.csv`**: Multivariate time series containing operational cycles of multiple aircraft engines. Each row includes 21 sensor readings (s1–s21), 3 settings (setting1–setting3), engine ID, and cycle number.
- **`component_rul.csv`**: Remaining Useful Life (RUL) data or engine degradation labels. Contains Unit_ID and remaining operational cycles before failure.

### Key Characteristics:
- Engines start with unknown initial wear and degradation patterns
- Degradation increases over operational cycles until a failure threshold is reached
- The final cycle for each engine marks its failure point (in training data)
- Multiple engines can have different degradation rates and failure modes

## Features

### Sensor Analysis
- **Trend Chart**: Displays rolling average of sensor s11 over operational cycles, highlighting detected anomalies (Z-score > 3)
- **Z-Score Histogram**: Distribution of normalized sensor values across all units, useful for identifying extreme degradation events
- **s11 Box Plot**: Compares baseline sensor values and variability across engine units

### RUL (Remaining Useful Life) Analysis
- **RUL Comparison Plot**: Overlay of true Remaining_Life vs. estimated RUL using a sliding-window linear regression on rolling averages
- **Anomaly Detection**: Flags HIGH ANOMALY events where Z-score exceeds 3, indicating potential critical degradation

### Reporting
- **Anomaly Table**: Lists all detected HIGH ANOMALY events for the selected unit with cycle number, sensor value, Z-score, and flag
- **Anomaly Pie Chart**: Proportion of normal vs. anomalous readings across the dataset

### Interactive Controls
- **Engine Unit Dropdown**: Select a specific engine to focus analysis on that unit's degradation trajectory
- Real-time filtering and dynamic plot updates

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone or download this repository:**
   ```bash
   cd "Aircraft Engine Predictive Maintenance Dashboard"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data:**
   - For `main_csv.py`: Place `raw_sensor_data.csv` and `component_rul.csv` in the project directory


## Usage

### Run CSV-Backed Dashboard (Recommended for Testing)
```bash
python main_csv.py
```
Then open your browser to `http://127.0.0.1:8050/`


## Visual Guide

### Trend Chart
Shows sensor degradation over time (rolling average of s11). 
- **Blue line**: Smoothed sensor reading trend
- **Red stars**: HIGH ANOMALY events (Z-score > 3)
- **Interpretation**: Rising trends indicate degradation; spikes may indicate imminent failure

### Z-Score Distribution Histogram
Displays the frequency of normalized sensor values.
- **Left tail (negative)**: Normal operation
- **Right tail (positive Z > 3)**: Anomalies and degradation events
- **Interpretation**: Higher concentration on the right = more engine degradation

### s11 Box Plot
Compares sensor s11 baseline across units.
- **Box**: Interquartile range (50% of data)
- **Whiskers**: 1.5× IQR
- **Points**: Outliers
- **Interpretation**: Units with higher median = higher baseline sensor readings

### RUL Comparison Plot
Overlays true Remaining_Life vs. estimated RUL.
- **Blue line**: Ground truth Remaining_Life from component_rul
- **Orange line**: Estimated RUL from sliding-window linear regression
- **Interpretation**: Close alignment = good model; divergence = needs tuning

### Anomaly Pie Chart
Proportion of HIGH ANOMALY vs. normal readings.
- **Large slice**: Few anomalies detected (healthy fleet)
- **Small slice**: Many anomalies (fleet experiencing degradation)

### Anomaly Table
Lists all detected anomaly events for the selected unit.
- **Operational_Cycle**: When the anomaly occurred
- **s11**: Sensor value at that cycle
- **Z_Score**: Normalized score (typically > 3 for HIGH ANOMALY)
- **Anomaly_Flag**: "HIGH ANOMALY" indicator

## Algorithm Details

### Anomaly Detection
1. **Rolling Average**: 5-cycle moving average smooths sensor noise
2. **Z-Score Normalization**: Per-unit standardization: `(x - mean) / std_dev`
3. **Threshold**: Values with Z-score > 3 flagged as HIGH ANOMALY

### RUL Estimation (Sliding-Window Approach)
1. **Window**: 10-cycle sliding window over rolling average
2. **Linear Fit**: Least-squares polyfit (degree 1) on sensor data
3. **Threshold**: Mean + 3×std of window values (failure boundary)
4. **Estimate**: `(threshold - current_value) / slope` = cycles to failure

**Note**: This is a simple estimator for demonstration. Production systems should use advanced ML models (LSTM, survival analysis, etc.).

## Customization

### Adjust Anomaly Threshold
In `main_csv.py`, line ~56:
```python
df_report['Anomaly_Flag'] = df_report['Z_Score'].apply(lambda v: 'HIGH ANOMALY' if pd.notna(v) and v > 3 else '')
```
Change `3` to a different Z-score threshold (e.g., `2.5` for more sensitivity).

### Change Rolling Window
In `main_csv.py`, line ~43:
```python
df_report['rolling_avg_s11'] = df_report.groupby('Unit_ID')['s11'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
```
Change `window=5` to adjust smoothing (larger = more smoothing).

### Modify RUL Estimator Window
In `main_csv.py`, inside `update_from_csv()` callback (~line 165):
```python
window = 10
```
Change `10` to adjust the estimation window size.

## Troubleshooting

### "No data for this unit" Message
- Check that CSV files are in the correct directory with the right column names
- Verify Unit_ID values in dropdown match the CSV data

### RUL Comparison Plot Shows Blank
- Ensure `Remaining_Life` column exists in `component_rul.csv`
- Verify `s11` column exists in `raw_sensor_data.csv`
- Check that selected unit has at least 3 rows of data with non-null s11 values

### Port 8050 Already in Use
Run on a different port:
```bash
python -c "from main_csv import app; app.run(port=8051, debug=True)"
```

## Project Structure

```
Aviation Component Anomaly Reporting Dashboard/
├── main_csv.py                # CSV-backed Dash application
├── raw_sensor_data.csv        # Sensor data (21 sensors, multiple units)
├── component_rul.csv          # RUL/remaining life labels
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── TODO.md                    # Project notes
```

```

## Dependencies

- **dash**: Interactive web framework
- **plotly**: Graphing library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

See `requirements.txt` for pinned versions.

## Performance Notes

- **CSV-backed app** (`main_csv.py`): Fast startup, suitable for < 1M rows
- **Dashboard response**: Sub-second interactivity for < 100k rows per unit

## Future Enhancements

- [ ] Advanced RUL models (LSTM, Random Forest)
- [ ] Multi-sensor anomaly detection (PCA, Isolation Forest)
- [ ] Predictive alerts and email notifications
- [ ] Data export/download (CSV, PDF reports)
- [ ] Tabbed interface for separate RUL and sensor analysis
- [ ] Real-time streaming data support

## License

This project is provided as-is for educational and research purposes. Modify and distribute freely.

## Contact & Support

For questions, issues, or suggestions, please check the project repository or contact the maintainers.

---

**Version**: 1.0  
**Last Updated**: November 2025
