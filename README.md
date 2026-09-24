# pytorch-realtime-stock-forecaster

A real-time stock forecasting pipeline built with Python and PyTorch. This initial version focuses on data scraping, ingestion, and preprocessing workflows.

## 📁 Repository Structure

Based on the project structure, here is the organizational layout:

```text
pytorch-realtime-stock-forecaster/
├── data/
│   ├── processed/             # Preprocessed feature binaries (.npy) [IGNORED]
│   │   └── .gitkeep           # Keeps directory in Git tracking
│   └── raw/                   # Raw historical datasets (.csv) [IGNORED]
│       ├── AAPL_raw.csv
│       ├── AMZN_raw.csv
│       └── TSLA_raw.csv
├── models/                    # Saved PyTorch weights and configurations
├── src/
│   ├── models/                # PyTorch architecture code (Linear Regression)
│   ├── automate_pipeline.py # Scheduling & execution engine
│   ├── main.py            # Main script entry point
│   └── scraper/               # Data collection & engineering scripts
│      ├── config.py          # Environment parameters & API keys
│      ├── parsers.py         # Response mapping and data cleaners
│      └── pipeline.py        # Core processing logic
|   
├── .gitignore                 # Exclusion configuration for data & caching
├── pyproject.toml             # Project dependency metadata
└── requirements.txt           # Package specifications
```

---

## ⚙️ Setup and Installation

### 1. Environment Activation
Ensure you are using the virtual environment configured for the project:
```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 2. Dependency Tracking
Install the necessary processing tools and libraries managed by your project configuration:
```bash
pip install -r requirements.txt
```

---

## 🛑 Git Tracking & Data Handling Workflow

The project is pre-configured to handle large local files cleanly. Raw historical CSV components and structural numpy arrays (`.npy`) are excluded from tracking to keep the remote repository small and performant.


## 🚀 Running the Data Pipeline

To trigger the basic fetching engine, execute the entry execution module:

```bash
python src/scraper/main.py
```

### Pipeline Lifecycle Tasks:
1. **Fetch & Extract**: Queries live markets to collect pricing history for tracking targets (`AAPL`, `AMZN`, `TSLA`).
2. **Raw Staging**: Saves continuous streams as `.csv` structures inside `data/raw/`.
3. **Array Quantization**: Processes time-series indicators into isolated features and targets, exporting binary scaling maps and data files (`*_X_train.npy`, `*_y_train.npy`) directly inside `data/processed/`.
