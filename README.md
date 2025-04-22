# Aircraft Collision Risk Analysis System

## Overview

This system implements an advanced solution for detecting, evaluating, and managing collision risks between aircraft using real-time ADS-B (Automatic Dependent Surveillance-Broadcast) data. The system processes aircraft position data, evaluates potential conflicts using multiple methodologies (fuzzy logic and Bayesian analysis), and generates specific recommendations to mitigate identified risks.

## Features

- **Comprehensive data preprocessing**: Cleans, normalizes and validates raw ADS-B data
- **Multi-factor priority classification**: Assigns priority levels to aircraft based on distance, speed, signal strength, and other parameters
- **Advanced risk assessment**: Combines fuzzy logic and Bayesian probability analysis
- **Passive monitoring**: Maintains tracking of medium-priority aircraft
- **Specific recommendation generation**: Produces actionable guidance based on risk level
- **Extensive visualization**: Generates visual representations for each analysis stage
- **Parallel processing**: Optimizes performance through batched parallel operations

## System Architecture

The system follows a sequential processing flow with six main modules:

1. **Data Preprocessing** - Normalizes and validates raw ADS-B data
2. **Priority Classification** - Assigns priority levels (0-9) to aircraft
3. **Fuzzy Logic Evaluation** - Calculates collision risk using fuzzy inference
4. **Bayesian Analysis** - Determines collision probabilities
5. **Passive Monitoring** - Tracks medium-priority aircraft
6. **Recommendation Generation** - Produces specific actions based on risk level

## Installation

```bash
# Clone the repository
git clone https://github.com/username/aircraft-collision-risk-analysis.git
cd aircraft-collision-risk-analysis

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run the complete analysis pipeline
python main.py

# Run individual modules
python -m src.preprocessing
python -m src.priority
python -m src.fuzzy_evaluation
python -m src.bayesian
python -m src.monitoring
python -m src.recommendations
```

### Configuration

The system configuration is defined in `src/config.py`. Key parameters include:

- `MAXDISTANCETHRESHOLD`: Maximum distance (km) to consider aircraft relevant
- `MINSPEEDTHRESHOLD`: Minimum speed (knots) to evaluate risk
- `DASHBOARDPORT`: Port for the visualization dashboard

## Data Flow

1. **Input**: Raw ADS-B data with fields including:
   - timestamp, addr (hex), flight, category
   - lat, lon, altitude (barometric and geometric)
   - speed (ground speed, IAS, TAS), track
   - and many other aircraft parameters

2. **Processing**: Sequential through all modules with intermediate results stored at each stage

3. **Output**: Comprehensive analysis with specific recommendations for each aircraft

## Module Descriptions

### Preprocessing Module

Cleans and normalizes raw ADS-B data, handling missing values, validating geographical coordinates, and standardizing formats. Generates spatial visualizations of aircraft distribution.

### Priority Classification Module

Implements a sophisticated scoring system that evaluates multiple factors to assign priority levels:
- Distance (up to 100 points)
- Speed (up to 80 points)
- RSSI/signal strength (up to 15 points)
- Wind (up to 6 points)
- Altitude (up to 15 points)
- Altitude change rate (up to 10 points)
- Turn rate (up to 5 points)

Complementary machine learning models (decision trees and gradient boosting) enhance classification.

### Fuzzy Logic Evaluation Module

Evaluates collision risk using a fuzzy inference system with four main linguistic variables:
- Distance (Near, Medium, Far)
- Speed difference (Negative, Low, Medium, High)
- Time to conflict (Moving Away, Short, Moderate, Long)
- Priority (Low, Medium, High)

Generates visualizations of decision surfaces and rule activation strengths.

### Bayesian Analysis Module

Calculates collision probabilities based on distance, relative velocity, and time to conflict using probability distributions. Combines probabilities using Bayesian theorem and categorizes aircraft into low, medium, and high-risk levels.

### Monitoring Module

Monitors medium-priority aircraft (3-5) and assigns states:
- Potential risk (distance  200 knots)
- Caution (distance < 20km)
- Stable (other cases)

Generates geospatial visualizations of monitoring states.

### Recommendation Generation Module

Produces specific recommendations based on previous analyses, using a weighted approach (60% Bayesian analysis, 40% fuzzy evaluation). Recommendations range from:
- "CRITICAL ALERT: Immediate evasive maneuver required"
- "Preventive maneuver suggested"
- "Active/passive monitoring"
- "Routine tracking"
- "Basic logging - No active tracking"

## Visualization Examples

The system generates various visualizations for each module:
- Spatial aircraft distribution maps
- Priority distribution charts
- Fuzzy decision surfaces
- Bayesian probability distributions
- Monitoring state maps
- Recommendation distribution charts

## Directory Structure

```
/
├── data/
│   ├── raw/             # Raw ADS-B data
│   ├── processed/       # Normalized data
│   ├── results/         # Analysis results
│   └── visualizations/  # Generated charts and maps
│       ├── preprocessing/
│       ├── priority/
│       ├── fuzzy_evaluation/
│       ├── bayesian/
│       ├── monitoring/
│       └── recommendations/
├── logs/                # System operation logs
├── src/
│   ├── config.py        # System configuration
│   ├── preprocessing.py # Data preprocessing
│   ├── priority.py      # Priority classification
│   ├── fuzzy_evaluation.py # Fuzzy logic evaluation
│   ├── bayesian.py      # Bayesian analysis
│   ├── monitoring.py    # Passive monitoring
│   ├── recommendations.py # Recommendation generation
│   └── utils/
│       ├── geometry.py  # Geometric calculations
│       ├── logger.py    # Logging configuration
│       └── data_loader.py # Data loading utilities
└── main.py              # Main execution script
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- simpful (for fuzzy logic)
- contextily (for map visualizations)
