# predictive-maintenance-classifier

A machine learning project built for as part of a Data Analysis coursework, taking on the role of a Business Analyst assigned to the maintenance team of a large manufacturing organisation. The goal is to predict machine failure types using historical sensor and operational data, enabling proactive maintenance scheduling and reducing unplanned downtime.


# Project Brief
"As a business analyst assigned to a maintenance team of a large manufacturing organization, assist in predicting common failures. The maintenance team has provided a historical dataset that contains characteristics tied to a given failure type and would like you to predict which failure will occur in the future."

The failure types include:
- No Failure (Machine is operating normally)
- Overstrain (Mechanical overload due to excessive torque or speed)
- Power Failures (Electrical/power-related failure)
Model Type: 3+ category prediction (multi-class classification)

# 📊 Dataset
The dataset contains 9,928 records of machine operational readings, each labelled with a failure outcome.

| Column | Data type | Description |
|---|---|---|
| `UDI` | int | Unique identifier — dropped during preprocessing |
| `Product_ID` | str | Product serial code — dropped during preprocessing |
| `Type` | str | Machine quality type: `L` (Low), `M` (Medium), `H` (High) |
| `Air_temperature_K_` | float | Ambient air temperature in Kelvin |
| `Process_temperature_K_` | float | Process temperature in Kelvin |
| `Rotational_speed_rpm_` | int | Rotational speed in revolutions per minute |
| `Torque_Nm_` | float | Torque applied in Newton-meters |
| `Tool_wear_min_` | int | Cumulative tool wear time in minutes |
| `Failure_Type` | str | **Target variable** — type of failure experienced |

# Models Used

1. Random Forest Classifier
2. Gradient Boosting Classifier

- Both models were trained on a **stratified 75/25 train-test split** (`random_state=42`) to preserve class proportions given the imbalanced target.
- Both models were chosen for their robust capability in handling structured tabular data and extracting non-linear relationships. The models' effectiveness is measured using comprehensive evaluation metrics, including accuracy score, classification reports, and confusion matrix visual displays.

# Key Findings

Based on feature importance scores from both models, the **top predictors of machine failure** are:

1. **Torque (Nm)** — highest predictive signal across both models
2. **Rotational Speed (rpm)** — strong indicator, especially for Power Failures
3. **Tool Wear (min)** — elevated wear strongly associated with Overstrain Failures

**Business Recommendation:** The maintenance team should prioritize continuous monitoring of torque sensors, rotational speed, and tool wear counters as the primary early-warning indicators.

---

## How to Run

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### 1. Clone or download the repository
```bash
git clone https://github.com/adedeji-adeboyejo/predictive-maintenance-classifier.git
cd predictive-maintenance
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> Note: Make sure 'requirements.txt' is in the same directory as the notebook before running.

### 3. Launch Jupyter Notebook
```bash
jupyter notebook predictive_classifier.ipynb
```

### 4. Run all cells

> Note: Make sure `maintenance_dataset.csv` is in the same directory as the notebook before running.
---

## Dependencies

See `requirements.txt`. Core libraries include:

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` & `seaborn` — data visualization
- `scikit-learn` — preprocessing, model training, and evaluation
---
