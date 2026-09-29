# Mobile Price Classification

Predictive pipeline classifying mobile phones into 4 price tiers based on hardware specifications (RAM, battery power, screen resolution, etc.).

---

## Data Pipeline & Architecture

* **Data Ingestion:** Loads dataset from a local MS SQL Server instance. If database connection is unavailable, it automatically falls back to local CSV files in the `data/` directory, allowing code execution without local DB setup.
* **Preprocessing:** Handles missing values via median imputation.
* **Modelling:** Trains a classification model and evaluates feature importance.
* **Outputs:** Automatically exports predictions to CSV and generates a summary plot.

---

## Project Structure

```text
mobile-price-classification/
├── data/
│   ├── train.csv                 # Training data (fallback)
│   ├── test.csv                  # Test data (fallback)
│   ├── test_predictions.csv      # Output predictions
│   └── test_predictions_plot.png # Output plot
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data ingestion (SQL / CSV fallback)
│   ├── preprocessing.py          # Data cleaning
│   ├── models.py                 # Model training and evaluation
│   └── visualization.py          # Plot generation
├── .env.example                  # Database configuration template
├── .gitignore
├── main.py                       # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Piotrem89/Mobile-price-classification.git
cd Mobile-price-classification
python -m venv .venv
```

Activate the environment:

* **Linux/macOS:** `source .venv/bin/activate`
* **Windows:** `.venv\Scripts\activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Pipeline

Execute the main script:

```bash
python main.py
```

---

## Generated Outputs

Upon completion, the pipeline outputs two files to the `data/` directory:

* `test_predictions.csv` – Predicted price categories for test records.
* `test_predictions_plot.png` – Visual summary of predicted class distribution and key feature importances.

> **Note:** By default, the script attempts to load data from the MS SQL Server database. If the database connection is unavailable, it automatically falls back to reading local CSV files from the `data/` directory.

