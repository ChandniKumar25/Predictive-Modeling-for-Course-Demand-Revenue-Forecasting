from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "CoursePrice", "CourseDuration", "CourseRating", "YearsOfExperience",
    "TeacherRating", "HistoricalEnrollmentCount", "PastAverageRevenue",
    "RevenuePerEnrollment", "ExpertiseCategoryMatch", "IsPaidCourse",
]
CATEGORICAL_FEATURES = [
    "CourseCategory", "CourseType", "CourseLevel", "Expertise",
    "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket",
]


def load_cleaned_data(data_path=None):
    path = Path(data_path or Path(__file__).parent / "data" / "edupro_merged_cleaned.csv")
    frame = pd.read_csv(path)
    for col in ["TransactionDate"]:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def add_live_features(input_values, frame):
    values = dict(input_values)
    price = float(values.get("CoursePrice", 0))
    duration = float(values.get("CourseDuration", frame["CourseDuration"].median()))
    rating = float(values.get("CourseRating", frame["CourseRating"].median()))
    experience = float(values.get("YearsOfExperience", frame["YearsOfExperience"].median()))
    category = str(values.get("CourseCategory", frame["CourseCategory"].mode().iat[0]))
    expertise = str(values.get("Expertise", frame["Expertise"].mode().iat[0]))
    history = frame.groupby("CourseCategory").agg(
        HistoricalEnrollmentCount=("TransactionID", "count"),
        PastAverageRevenue=("Amount", "mean"),
    )
    category_history = history.loc[category] if category in history.index else history.mean(numeric_only=True)
    result = {
        "CoursePrice": price,
        "CourseDuration": duration,
        "CourseRating": rating,
        "YearsOfExperience": experience,
        "TeacherRating": float(values.get("TeacherRating", frame["TeacherRating"].median())),
        "HistoricalEnrollmentCount": float(category_history["HistoricalEnrollmentCount"]),
        "PastAverageRevenue": float(category_history["PastAverageRevenue"]),
        "RevenuePerEnrollment": float(category_history["PastAverageRevenue"] / max(category_history["HistoricalEnrollmentCount"], 1)),
        "ExpertiseCategoryMatch": int(expertise.lower() == category.lower()),
        "IsPaidCourse": int(price > 0),
        "CourseCategory": category,
        "CourseType": str(values.get("CourseType", "Paid" if price > 0 else "Free")),
        "CourseLevel": str(values.get("CourseLevel", frame["CourseLevel"].mode().iat[0])),
        "Expertise": expertise,
        "PriceBand": "Free" if price == 0 else "Budget" if price <= 100 else "Standard" if price <= 300 else "Premium",
        "DurationBucket": "Short" if duration <= 10 else "Medium" if duration <= 25 else "Long" if duration <= 45 else "Extended",
        "RatingTier": "Low" if rating <= 2.5 else "Fair" if rating <= 3.5 else "Good" if rating <= 4.25 else "Excellent",
        "ExperienceBucket": "Early Career" if experience <= 2 else "Developing" if experience <= 5 else "Experienced" if experience <= 10 else "Veteran",
    }
    return pd.DataFrame([result])
