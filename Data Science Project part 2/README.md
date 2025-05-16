
# ✅ Data Science Project Steps Summary

This document outlines all the steps taken in each of the four data science tasks included in the project.

---

## ✅ Task 1: Employee Attrition Prediction
**Dataset**: IBM HR Analytics Dataset

### 🔍 1. Exploratory Data Analysis (EDA)
- [x] Loaded dataset and checked shape, data types
- [x] Checked missing values and duplicates
- [x] Univariate analysis (Attrition, Age, Department, Job Role, etc.)
- [x] Bivariate analysis (Attrition vs. other features using barplots and boxplots)
- [x] Correlation matrix and heatmap
- [x] Feature engineering (e.g., encoding categorical variables)

### ⚙️ 2. Model Training
- [x] Split data into training and test sets
- [x] Trained Logistic Regression model
- [x] Trained Random Forest classifier
- [x] Compared models using Accuracy, Precision, Recall, and F1 Score

### 📊 3. Explainability
- [x] Applied SHAP values for feature importance
- [x] Visualized top features affecting attrition
- [x] Used LIME for local interpretability

### 🧠 4. Insights & Recommendations
- [x] Identified key drivers of attrition (e.g., OverTime, JobSatisfaction)
- [x] Suggested actionable HR strategies for retention

---

## ✅ Task 2: Text Summarization
**Dataset**: CNN/DailyMail from Hugging Face

### 🧹 1. Preprocessing
- [x] Loaded dataset using `datasets` library
- [x] Lowercased text, removed special characters
- [x] Tokenized text using `nltk` and `spaCy`

### 📌 2. Extractive Summarization
- [x] Used spaCy to extract key sentences
- [x] Calculated sentence scores based on word frequency
- [x] Selected top sentences as the summary

### 🔁 3. Abstractive Summarization
- [x] Used Hugging Face's `transformers` to load T5/BART
- [x] Generated summaries using pre-trained model
- [x] Compared results with reference summaries

### 🧪 4. Fine-tuning & Evaluation
- [x] Fine-tuned model on CNN/DailyMail training data
- [x] Evaluated using ROUGE scores
- [x] Validated summary coherence manually

---

## ✅ Task 3: Disease Diagnosis Prediction
**Dataset**: PIMA Diabetes Dataset or Heart Disease Dataset

### 🔍 1. EDA
- [x] Loaded dataset and examined distributions
- [x] Checked for missing values and outliers
- [x] Analyzed feature relationships with the target (disease)

### ⚙️ 2. Preprocessing
- [x] Normalized or scaled numeric features
- [x] Performed feature selection (e.g., based on correlation or feature importance)

### 🤖 3. Model Training
- [x] Trained Gradient Boosting Classifier
- [x] Trained Support Vector Machine (SVM)
- [x] Trained a simple Neural Network
- [x] Evaluated using F1 Score and AUC-ROC curve

### 💡 4. Insights
- [x] Identified top predictive features (e.g., Glucose, BMI)
- [x] Provided early-warning recommendations for patients

---

## ✅ Task 4: Loan Default Prediction
**Dataset**: Lending Club Loan Dataset (From OpenIntro)

### 🧹 1. Data Preprocessing
- [x] Loaded data in chunks due to size
- [x] Handled missing values
- [x] Encoded categorical variables
- [x] Handled class imbalance using SMOTE

### ⚙️ 2. Model Training
- [x] Trained LightGBM Classifier
- [x] Trained SVM for comparison
- [x] Tuned hyperparameters with GridSearchCV

### 📊 3. Model Evaluation
- [x] Evaluated with Precision, Recall, F1 Score
- [x] Plotted confusion matrix and ROC Curve

### 📈 4. Final Report
- [x] Listed best-performing model and metrics
- [x] Provided risk indicators and lending recommendations

---

### 📈 DATASETS

You can download it here: [Download from Google Drive](https://drive.google.com/file/d/1Ek5nEX3E4xPH5Xq4s9UpJleIrERKAkna/view?usp=drive_link)
