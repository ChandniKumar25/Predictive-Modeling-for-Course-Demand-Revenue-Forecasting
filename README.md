# EduPro Predictive Analytics

Premium Streamlit application for predictive course demand and revenue forecasting.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app uses `data/edupro_merged_cleaned.csv`, benchmarks Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting for enrollment, course revenue, and category revenue, then deploys the lowest-validation-RMSE model for live predictions.
