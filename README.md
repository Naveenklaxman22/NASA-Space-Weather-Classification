# NASA-Space-Weather-Classification
Machine learning project classifying solar flare severity from NASA space weather data, featuring imbalance handling and Power BI visualization.

# NASA Space Weather Data Analysis: Solar Flare Classification

## Table of Contents
- [1. Overview](#1-overview)
- [2. Data Source](#2-data-source)
- [3. Methodology](#3-methodology)
    - [3.1 Data Cleaning & Preprocessing](#31-data-cleaning--preprocessing)
    - [3.2 Exploratory Data Analysis (EDA)](#32-exploratory-data-analysis-eda)
    - [3.3 Model Selection & Training](#33-model-selection--training)
    - [3.4 Model Evaluation](#34-model-evaluation)
- [4. Key Findings & Actionable Insights](#4-key-findings--actionable-insights)
- [5. Interactive Dashboard (Power BI)](#5-interactive-dashboard-power-bi)
- [6. Tools Used](#6-tools-used)
- [7. Future Work](#7-future-work)
- [8. Contact Information](#8-contact-information)

---

## 1. Overview

This project delves into **NASA Space Weather Data Analysis**, focusing on **Solar Flare Event Analysis & Severity Classification**. The primary goal was to build a machine learning model to predict the severity class (C, M, X) of solar flares, which are crucial phenomena impacting Earth's technological infrastructure. This project navigates complex scientific data, addresses extreme class imbalance, and provides insights through a comprehensive analytical pipeline culminating in a Power BI dashboard.

---

## 2. Data Source

The dataset used is the **"NASA_space_weather_data"** from Kaggle, which is a collection of various space weather event files from NASA's DONKI API. For this specific project, the **`solar_flares.csv`** file was chosen as the primary data source due to its direct relevance to solar flare events and classifications.

* **Link to Kaggle Dataset:** [NASA Space Weather Data](https://www.kaggle.com/datasets/edacelikeloglu/nasa-space-weather-data)
* **Primary File Used:** `solar_flares.csv` (Part of the larger dataset archive)

---

## 3. Methodology

My analytical approach involved a structured machine learning pipeline:

### 3.1 Data Cleaning & Preprocessing

* **Initial Acquisition:** Loaded `solar_flares.csv` (1599 entries, 10 columns).
* **Column Dropping:** Removed irrelevant or highly null columns: `event_id`, `event_type` (constant value), `note` (too many NaNs), `source_location` (too high cardinality).
* **Datetime Conversion & Feature Engineering:** Converted `begin_time`, `peak_time`, `end_time` to datetime objects and calculated `event_duration_minutes`.
* **Missing Value Handling:** Dropped rows with any remaining NaNs (primarily `active_region`), reducing the dataset from 1599 to 1552 rows.
* **Target Encoding (`class_type`):**
    * Extracted the main class character (A, B, C, M, X) into `main_class`.
    * **Addressed Extreme Imbalance:** Crucially, **filtered out 'A' and 'B' classes** (which had fewer than 2 samples) to enable stratified splitting. The target classes for classification became 'C', 'M', and 'X'.
    * Applied `OrdinalEncoder` to map 'C'->0, 'M'->1, 'X'->2.
* **Feature Identification:** Identified `active_region`, `event_duration_minutes` (numerical) and `instruments` (categorical) as final features.
* **One-Hot Encoding:** Applied `pd.get_dummies` to `instruments` column.
* **Feature Scaling:** Applied `StandardScaler` to numerical features (`active_region`, `event_duration_minutes`).
* **Column Name Sanitization:** Cleaned feature names for compatibility with LightGBM.
* **Final Data Shape:** `X_processed` (features) has `(1551, 3)` shape, and `y` (target) has `(1551,)` shape.

### 3.2 Exploratory Data Analysis (EDA)

Comprehensive EDA was performed to understand the dataset's characteristics and challenges:

* **Target Class Distribution:** Confirmed **extreme class imbalance**, with 'M' class dominant, and 'C', 'X' as minorities. This highlighted the primary challenge for classification.
    ![Solar Flare Class Distribution](plots/[YOUR_SOLAR_FLARE_CLASS_DISTRIBUTION_PLOT_FILENAME].png)
* **Feature Distributions by Class:** Histograms for `active_region` and `event_duration_minutes` showed considerable overlap across classes, suggesting weak individual discriminative power.
    ![Numerical Feature Distributions by Class](plots/[YOUR_NUMERICAL_FEATURE_DISTRIBUTIONS_PLOT_FILENAME].png)
* **Instrument Distribution:** Visualized the dominance of one instrument type, implying limited predictive power for the `instruments` feature.
    ![Instruments by Solar Flare Class](plots/[YOUR_INSTRUMENTS_BY_CLASS_PLOT_FILENAME].png)
* **Correlation Matrix:** Revealed weak to moderate correlations between features and the encoded class type, indicating a complex classification problem.
    ![Correlation Matrix Heatmap](plots/[YOUR_CORRELATION_MATRIX_PLOT_FILENAME].png)

### 3.3 Model Selection & Training

* **Data Split:** Data was split into training (80%) and testing (20%) sets using **stratified sampling** to maintain class proportions, crucial for imbalanced data.
* **Model Choices:** Evaluated three robust classification models:
    * **Logistic Regression:** As a baseline.
    * **Random Forest Classifier:** For its robustness.
    * **LightGBM Classifier:** For its efficiency and performance.

### 3.4 Model Evaluation

The models' performance revealed the significant challenges posed by the extreme class imbalance and limited features.

* **Key Metrics (LightGBM Classifier - Example from your run):**
    * **Accuracy:** `[YOUR_ACCURACY_VALUE]` (e.g., 0.7942)
    * **Classification Report (Crucial for Imbalance):**
        * **Class C (e.g., 0.0):** Precision `[YOUR_C_PRECISION]`, Recall `[YOUR_C_RECALL]`, F1-score `[YOUR_C_F1]` (e.g., C: P 0.28, R 0.15, F1 0.20)
        * **Class M (e.g., 1.0):** Precision `[YOUR_M_PRECISION]`, Recall `[YOUR_M_RECALL]`, F1-score `[YOUR_M_F1]` (e.g., M: P 0.86, R 0.91, F1 0.88)
        * **Class X (e.g., 2.0):** Precision `[YOUR_X_PRECISION]`, Recall `[YOUR_X_RECALL]`, F1-score `[YOUR_X_F1]` (e.g., X: P 0.08, R 0.07, F1 0.08)
    * **Weighted ROC AUC Score (OVR):** `[YOUR_ROC_AUC_VALUE]` (e.g., 0.6813)
* **Interpretation:** While overall accuracy might seem moderate, the **low Precision, Recall, and F1-scores for minority classes ('C' and 'X')** show that models largely struggled to distinguish these rare events, often defaulting to the majority 'M' class. LightGBM was the best performing model, showing a slight edge in predicting minority classes, but overall performance across all classes remained challenging due to the data's inherent properties.

---

## 4. Key Findings & Actionable Insights

The feature importance analysis from the LightGBM model provided insights into what the model prioritized:

![Solar Flare Feature Importances](plots/[YOUR_FEATURE_IMPORTANCE_PLOT_FILENAME].png)

* **`active_region`** was identified as the most important feature (Importance: `[YOUR_ACTIVE_REGION_IMPORTANCE]`).
* **`event_duration_minutes`** was the second most important (Importance: `[YOUR_EVENT_DURATION_IMPORTANCE]`).
* The `instruments` feature contributed very little.

**Actionable Recommendations (considering challenges):**
* **Focus on Source & Duration:** Initial predictions can leverage `active_region` and `event_duration_minutes` as key indicators.
* **Recognize Limitations:** Acknowledge that robust, fine-grained multi-class classification of solar flares is very challenging with limited, non-image-based features and extreme class imbalance.
* **Future Research:** More advanced techniques like deep learning on solar image data, specialized imbalance handling algorithms (e.g., SMOTE-NC, cost-sensitive learning), or collecting more discriminative data would be needed for a highly reliable real-time prediction system.
* **Impact on Mitigation:** Despite classification challenges, identifying any potential for X-class flares (even if recall is low) is critical for early warnings to protect infrastructure.

---

## 5. Interactive Dashboard (Power BI)

An interactive Power BI dashboard was developed to visually summarize the project's findings, especially highlighting the challenges of multi-class imbalance.

* **Purpose:** To present complex classification results and model limitations in an intuitive format for stakeholders.
* **Key Visuals:** Includes a Class Distribution chart, KPI cards for overall accuracy and per-class Precision/Recall/F1-score, a visual Confusion Matrix, and a Feature Importance chart.
* **Dashboard Overview Example:**
    ![Dashboard Overview](plots/[YOUR_DASHBOARD_OVERVIEW_FILENAME].png)
* **Performance Metrics Example:**
    ![Dashboard Metrics](plots/[YOUR_DASHBOARD_METRICS_FILENAME].png)
* **Confusion Matrix Example:**
    ![Dashboard Confusion Matrix](plots/your_dashboard_confusion_matrix.png)

---

## 6. Tools Used

* `Python`
* `Pandas` (for data manipulation)
* `NumPy` (for numerical operations)
* `Scikit-learn` (for preprocessing, model selection, and evaluation)
* `LightGBM` (for gradient boosting classification)
* `Matplotlib` & `Seaborn` (for data visualization in Python)
* `Power BI` (for interactive dashboarding and reporting)
* **Conceptual SQL:** (In a real-world scenario, initial data extraction and aggregation from a larger database would be performed using SQL.)

---

## 7. Future Work

* **Advanced Feature Engineering:** Explore deriving more complex features from solar flare data (e.g., analyzing patterns of `active_region` evolution over time, or using image data if available).
* **Imbalance Handling Techniques:** Implement more sophisticated methods like oversampling (e.g., SMOTE-NC), undersampling, or cost-sensitive learning specifically for multi-class imbalanced datasets.
* **Model Interpretability:** Deepen the understanding of misclassifications for minority classes to identify patterns that lead to incorrect predictions.

---

## 8. Contact Information

* **Naveen Lakshman Kumar Basina**
* **LinkedIn:** [https://www.linkedin.com/in/naveen-lakshman/](https://www.linkedin.com/in/naveen-lakshman/)
* **Email:** [naveenklaxman22@gmail.com](mailto:naveenklaxman22@gmail.com)
