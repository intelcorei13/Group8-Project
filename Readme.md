# Loan Approval — Supervised Learning Project 

## Author
- **Christian Sekpe** 


---

## Overview
This project builds an interactive **Streamlit** web app to predict **loan approval** using supervised learning.  
We compare a **Decision Tree Classifier** and **Gaussian Naive Bayes (GNB)**, and provide clear visualizations, model metrics, and a quick “what-if” prediction page.  

The app also demonstrates:
- Exploratory Data Analysis (EDA) with charts and distributions  
- Feature engineering (including EMI creation)  
- Preprocessing with encoding & scaling  
- Model training and threshold tuning  
- Cross-validation (10-fold) for robust evaluation  
- Side-by-side comparison of models with feature importance  

---

## Features
- **Dataset Page** — View dataset  
- **EDA Page** — Explore distributions, correlations, and relationships  
- **Classification: Decision Tree** — Train, tune thresholds, view feature importance  
- **Classification: Naive Bayes** — Train with scaled features, view importance via permutation  
- **Prediction Page** — Enter applicant details and get real-time predictions  
- **Interpretation & Conclusions** — Compare DT vs NB with metrics and feature insights  

---

## Models & Features
- **Decision Tree (DT):** `HasGoodCredit, TotalIncome, EMI, LoanAmount, Property_Area, IsMarried`  
- **Naive Bayes (NB):** `HasGoodCredit, TotalIncome, EMI, LoanAmount, Property_Area, IsMarried, Dependents`  

**Engineered Feature:**  
- **EMI** = LoanAmount ÷ Loan_Amount_Term (Equated Monthly Installment)  

---

## Results
- **Decision Tree:** More interpretable, highlights key drivers, flexible threshold adjustment.  
- **Naive Bayes:** Simpler, robust with preprocessing and one-hot encoding, competitive performance.  
- **Trade-offs:** DT provides clearer interpretability; NB works well with scaled data.  

---

## Screenshot
Here’s a preview of the app interface (Dataset Page):

![Loan Approval App Screenshot](assets/dataset_page.png)

---

## How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/intelcorei13/Loan-Approval-Prediction
   ```
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app locally  
   ```bash
   streamlit run LoanApprovalPredictor.py
   ```

---

## Live Demo
- 🌍 **Streamlit App:** [loan-approval-prediction-website.streamlit.app](https://loan-approval-prediction-website.streamlit.app)  
- 💻 **GitHub Repo:** [github.com/intelcorei13/loan-approval-prediction](https://github.com/intelcorei13/Loan-Approval-Prediction)  

---

## Acknowledgment
This project was developed as part of a **Supervised Machine Learning Project** at the **University of Ghana Business School, OMIS DEPT**.  
