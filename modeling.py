import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES

TARGETS = {
    "Enrollment Count": "EnrollmentCount",
    "Course Revenue": "CourseRevenue",
    "Category Revenue": "CategoryRevenue",
}


def prepare_training_frame(frame):
    work = frame.copy()
    work["EnrollmentCount"] = 1
    course_revenue = work.groupby("CourseID")["Amount"].transform("sum")
    category_revenue = work.groupby("CourseCategory")["Amount"].transform("sum")
    work["CourseRevenue"] = course_revenue
    work["CategoryRevenue"] = category_revenue
    return work


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def model_candidates():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=180, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=140, learning_rate=0.05, max_depth=3, random_state=42),
    }


def train_all_models(frame, test_size=0.2):
    work = prepare_training_frame(frame)
    features = work[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    results = {}
    for target_name, target_col in TARGETS.items():
        y = work[target_col].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=test_size, random_state=42)
        model_results = []
        fitted = {}
        for model_name, estimator in model_candidates().items():
            pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            metrics = {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "R2 Score": r2_score(y_test, predictions),
            }
            model_results.append(metrics)
            fitted[model_name] = pipeline
        score_table = pd.DataFrame(model_results).sort_values(["RMSE", "MAE"], ascending=True).reset_index(drop=True)
        best_name = score_table.iloc[0]["Model"]
        results[target_name] = {"table": score_table, "models": fitted, "best_name": best_name, "best_model": fitted[best_name]}
    return results


def feature_importance_table(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_)).ravel()
    else:
        importances = np.zeros(len(feature_names))
    table = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    table["Feature"] = table["Feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    return table.sort_values("Importance", ascending=False).head(12).reset_index(drop=True)
