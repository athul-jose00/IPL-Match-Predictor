# 🏏 IPL Match Predictor

A machine learning project that predicts the probability of winning for IPL cricket matches during the second innings using various classification algorithms.

## 📊 Project Overview

This project analyzes IPL match data to predict match outcomes based on real-time match situations. The model considers factors like current score, target, overs remaining, wickets fallen, and team performance to provide win probability predictions.

## 🎯 Features

- **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, and Neural Networks
- **Real-time Predictions**: Calculates win probability during second innings
- **Interactive Web App**: Streamlit-based user interface for live predictions
- **Comprehensive Analysis**: Detailed EDA and feature engineering
- **Model Comparison**: Performance evaluation across different algorithms

## 📈 Dataset

The project uses two main datasets:

- `matches.csv`: Contains match-level information (teams, venue, winner, etc.)
- `deliveries.csv`: Contains ball-by-ball data for each match

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/[username]/ipl-match-predictor.git
cd ipl-match-predictor

# Install required packages
pip install pandas numpy seaborn matplotlib scikit-learn xgboost tensorflow streamlit
```

## 🚀 Usage

### Running the Jupyter Notebook

```bash
jupyter notebook MathcPredictor.ipynb
```

### Running the Streamlit App

```bash
streamlit run app1.py
```

## 🧠 Machine Learning Models

| Model               | Accuracy | Description                                    |
| ------------------- | -------- | ---------------------------------------------- |
| Logistic Regression | ~85%     | Linear classification with probability outputs |
| Decision Tree       | ~82%     | Tree-based classification                      |
| Random Forest       | ~87%     | Ensemble of decision trees                     |
| SVM                 | ~84%     | Support Vector Machine with RBF kernel         |
| XGBoost             | ~88%     | Gradient boosting classifier                   |
| Neural Network      | ~86%     | Deep learning with 2 hidden layers             |

## 📊 Key Features Used

- **runs_left**: Runs required to win
- **balls_left**: Balls remaining in the innings
- **wickets**: Wickets remaining
- **cur_run_rate**: Current run rate
- **req_run_rate**: Required run rate
- **batting_team**: Team currently batting
- **bowling_team**: Team currently bowling
- **city**: Match venue

## 🎮 Web Application

The Streamlit web app allows users to:

- Select batting and bowling teams
- Input current match situation (score, overs, wickets)
- Choose from multiple ML models
- Get real-time win probability predictions

### App Features:

- Team selection dropdown
- Real-time score input
- Model selection (Logistic Regression, SVM, Neural Network)
- Probability visualization

## 📁 Project Structure

```
ipl-match-predictor/
│
├── MathcPredictor.ipynb    # Main analysis notebook
├── app1.py                 # Streamlit web application
├── README.md              # Project documentation
├── models/                # Saved model files
│   ├── logistic_model.pkl
│   ├── svc_model.pkl
│   ├── model_ann.h5
│   ├── scaler.pkl
│   └── encoded_columns.pkl
└── data/                  # Dataset files
    ├── matches.csv
    └── deliveries.csv
```

## 🔍 Data Analysis Highlights

- **Match Distribution**: Analysis of matches played across different cities
- **Team Performance**: Win statistics for all IPL teams
- **Score Distribution**: First innings score patterns
- **Feature Correlation**: Relationship between features and match outcomes

## 🎯 Model Performance

The models show strong predictive capability with:

- High accuracy across different algorithms
- Good correlation between required run rate and match outcome
- Effective handling of categorical variables through one-hot encoding
- Proper scaling for distance-based algorithms

## 🙏 Acknowledgments

- IPL for providing the cricket data
- Scikit-learn and TensorFlow communities
- Streamlit for the web framework
- Cricket analytics community for inspiration



---

**Note**: This project is for educational and research purposes. Predictions are based on historical data and should not be used for betting or gambling purposes.
