<p align="center">
  <img src="https://img.icons8.com/3d-fluency/94/hospital.png" width="80"/>
</p>

<h1 align="center">🏥 ML-Healthcare-Diagnostics</h1>

<p align="center">
  <em>End-to-end Machine Learning project — from raw data exploration to a production-ready interactive dashboard — for early-stage healthcare screening.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg"/>
</p>

---

## 📌 Project Overview

This project applies both **Supervised** and **Unsupervised** Machine Learning techniques to two real-world healthcare datasets, covering the complete data science lifecycle:

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

| Module | Task | Technique |
|--------|------|-----------|
| **Module 1** | Diabetes Prediction | Supervised Classification |
| **Module 2** | Heart Patient Segmentation | Unsupervised Clustering |
| **Deployment** | Interactive Dashboard | Streamlit Web App |

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only** and must not replace professional medical diagnosis.

---

## 🗂️ Project Structure

```
ML-Healthcare-Diagnostics/
│
├── Supervised Model/
│   └── Supervised_ML.ipynb         # Full supervised learning pipeline
│
├── Unsupervised Model/
│   └── Unsupervised_ML.ipynb       # Full unsupervised learning pipeline
│
├── Datasets/
│   ├── diabetes.csv                # Pima Indians Diabetes Dataset (768 × 9)
│   └── heart.csv                   # UCI Heart Disease Dataset (1025 × 14)
│
├── models/                         # Serialised model artifacts (.pkl)
│   ├── diabetes_model.pkl          #   Random Forest Classifier
│   ├── scaler.pkl                  #   StandardScaler (heart features)
│   ├── pca.pkl                     #   PCA transformer (heart features)
│   └── kmeans_model.pkl            #   K-Means model (3 clusters)
│
├── Plots/                          # Saved visualisations from training
│
├── healthcare_diagnostics_app.py   # 🚀 Streamlit deployment app
├── requirements.txt                # Python dependencies
└── README.md
```

---

## ⚙️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn (Classification, Clustering, PCA, Scaling) |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Deployment** | Streamlit |
| **Serialisation** | Joblib |

---

## 📊 Module 1 — Diabetes Prediction (Supervised Learning)

> **Notebook:** `Supervised Model/Supervised_ML.ipynb`  
> **Dataset:** Pima Indians Diabetes Dataset — 768 patients, 8 clinical features, binary target (`Outcome`)

### 1.1 Exploratory Data Analysis (EDA)

- Inspected data shape, types, summary statistics, unique values
- Identified **biologically invalid zero values** in Glucose, BloodPressure, SkinThickness, Insulin, and BMI
- Visualised **class distribution** (Outcome: Diabetic vs Non-Diabetic)
- Plotted **histograms** of key features (Glucose, BMI, Age)
- Generated **correlation heatmap** to identify inter-feature relationships
- Created **box plots** for outlier detection

### 1.2 Data Preprocessing

| Step | Details |
|------|---------|
| **Invalid Zero Replacement** | Replaced 0s in Glucose, BloodPressure, SkinThickness, Insulin, BMI with column **median** values |
| **Missing Values** | Verified no NaN values present |
| **Feature Scaling** | Applied `StandardScaler` to all 8 features; verified mean ≈ 0, std ≈ 1 |
| **Categorical Encoding** | Checked for object-type columns (none found — all numeric) |

### 1.3 Feature Engineering

- Created **AgeGroup** bins: 21-30, 31-40, 41-50, 51-60, 61-70, 71+
- Created **BMICategory**: Underweight, Normal, Overweight, Obese
- Applied **One-Hot Encoding** (`drop_first=True`) on engineered categorical features
- Expanded feature set from **8 → 16+** columns

### 1.4 Train/Test Split

- **80/20** stratified split using `train_test_split(random_state=42)`

### 1.5 Models Trained & Compared

Four classifiers were trained and evaluated on the same test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Decision Tree** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Logistic Regression** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Random Forest** ⭐ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Support Vector Machine (SVM)** | ✅ | ✅ | ✅ | ✅ | ✅ |

> *Full metric values are available inside the notebook. The **Random Forest** was selected as the final model and saved to `models/diabetes_model.pkl`.*

Each model includes:
- **Classification Report** (Precision, Recall, F1, ROC-AUC)
- **Confusion Matrix** heatmap
- **Bar chart** comparing all models side-by-side across all 5 metrics

### 1.6 Model Explainability (SHAP)

- Used **SHAP TreeExplainer** on the Random Forest
- Generated **SHAP summary bar plot** to identify the most influential features for predicting diabetes
- Ranked features by importance using both **Random Forest `.feature_importances_`** and **SHAP values**

### 1.7 Model Serialisation

```python
joblib.dump(rf, 'models/diabetes_model.pkl')
```

---

## 📊 Module 2 — Heart Patient Segmentation (Unsupervised Learning)

> **Notebook:** `Unsupervised Model/Unsupervised_ML.ipynb`  
> **Dataset:** UCI Heart Disease Dataset — 1,025 patients, 13 features, binary target (`target`)

### 2.1 Data Preprocessing

| Step | Details |
|------|---------|
| **Missing Values** | Dropped rows with NaN values |
| **Feature Scaling** | Applied `StandardScaler` on numeric columns: `age`, `trestbps`, `chol`, `thalach`, `oldpeak` |
| **One-Hot Encoding** | Encoded multi-class categoricals: `cp`, `restecg`, `slope`, `thal` (with `drop_first=True`) |
| **Outlier Removal** | Removed outliers using **Z-score threshold = 2.5** on numeric columns |

### 2.2 Feature Selection

- Applied **SelectKBest** with `f_classif` scoring function
- Selected **top 10 features** from the expanded feature set
- Reduced noise and improved clustering quality

### 2.3 Train/Test Split

- **80/20** stratified split using `train_test_split(stratify=y, random_state=42)`

### 2.4 Dimensionality Reduction — PCA

| PCA Stage | Components | Purpose |
|-----------|-----------|---------|
| **Stage 1** | `n_components=2` | For 2D scatter plot visualisation |
| **Stage 2** | `n_components=0.95` | Retain 95% variance for boosted clustering |

### 2.5 Clustering — K-Means

#### Optimal k Selection

- **Elbow Method** — Plotted inertia vs. number of clusters (k = 1 to 10)
- **Silhouette Score** — Evaluated cluster cohesion and separation (k = 2 to 10)
- Selected **k = 3** as the optimal number of clusters

#### Final Model

```python
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
```

### 2.6 Visualisation

- **PCA Scatter Plot** — 2D projection of clusters coloured by cluster label using Seaborn
- Each cluster represents a distinct cardiovascular risk profile

### 2.7 Boosted Clustering with PCA (95% Variance)

- Re-ran PCA retaining **95% of total variance**
- Re-evaluated with **Silhouette Score** to measure clustering improvement

### 2.8 Model Serialisation

```python
joblib.dump(scaler, 'scaler.pkl')          # StandardScaler
joblib.dump(pca, 'pca.pkl')                # PCA transformer
joblib.dump(kmeans_final, 'kmeans_model.pkl')  # K-Means (3 clusters)
```

---

## 🖥️ Module 3 — Streamlit Deployment

> **File:** `healthcare_diagnostics_app.py`

A polished, production-ready web dashboard that loads the pre-trained models and exposes both modules as interactive tools for healthcare professionals.

### App Features

| Feature | Description |
|---------|-------------|
| 🩸 **Diabetes Prediction Page** | Form with sliders & number inputs → Random Forest prediction + probability score |
| ❤️ **Heart Segmentation Page** | Full UCI feature form → Scaler → PCA → K-Means cluster assignment |
| 🎨 **Modern UI** | Gradient headers, styled result cards (green/yellow/red), 2-column responsive layout |
| 📖 **Model Explainers** | Expandable `st.expander` sections explaining the ML pipeline behind each module |
| 🛡️ **Error Handling** | Graceful warnings if `.pkl` files are missing — app never crashes |
| ⚡ **Cached Loading** | `@st.cache_resource` loads models once for fast repeat predictions |

### Diabetes Page Output
- ✅ **Low Risk** (green card) or 🚨 **High Risk** (red card)
- Confidence probability with progress bar
- Clinical recommendation

### Heart Segmentation Output
- 🟢 **Cluster 0 — Low Risk Profile**
- 🟡 **Cluster 1 — Moderate Risk / At-Risk**
- 🔴 **Cluster 2 — High Risk Profile**
- Expandable patient data summary

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9** or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ML-Healthcare-Diagnostics.git
   cd ML-Healthcare-Diagnostics
   ```

2. **Create a virtual environment** *(recommended)*
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run healthcare_diagnostics_app.py
   ```

5. Open your browser at **http://localhost:8501** 🎉

> 🌐 **Or try the live demo:** [ML-Healthcare-Diagnostics on Streamlit Cloud](https://ml-healthcare-diagnostics-flndemsjreipr4znywjg4f.streamlit.app)

### Re-running the Notebooks

To reproduce the model training, open the notebooks in Jupyter or VS Code and install the full data science stack:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap scipy joblib
```

---

## 📂 Datasets

| Dataset | Source | Samples | Features | Target |
|---------|--------|---------|----------|--------|
| **Pima Indians Diabetes** | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) | Outcome (0/1) |
| **UCI Heart Disease** | [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) | 1,025 | 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) | target (0/1) |

---

## 🧠 Key Concepts Demonstrated

- ✅ Exploratory Data Analysis (EDA) & Data Cleaning
- ✅ Handling invalid zero values via median imputation
- ✅ Feature Engineering (binning, one-hot encoding)
- ✅ Feature Scaling (StandardScaler)
- ✅ Feature Selection (SelectKBest with ANOVA F-test)
- ✅ Supervised Classification (Decision Tree, Logistic Regression, Random Forest, SVM)
- ✅ Model Comparison (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Model Explainability (SHAP values)
- ✅ Dimensionality Reduction (PCA)
- ✅ Unsupervised Clustering (K-Means)
- ✅ Cluster Validation (Elbow Method, Silhouette Score)
- ✅ Outlier Detection & Removal (Z-score)
- ✅ Model Serialisation (Joblib)
- ✅ Web App Deployment (Streamlit)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Abdulrahman**

---

<p align="center">
  Made with ❤️ for better healthcare diagnostics
</p>
