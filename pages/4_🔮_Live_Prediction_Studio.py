import numpy as np
import streamlit as st
from common import DATA, MODELS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, add_live_features, init_page, money, safe_options

init_page('04 · Live prediction studio', 'Configure a course scenario')
st.caption('Adjust the inputs and the deployed models recalculate the forecast immediately.')
with st.container(border=True):
    row1 = st.columns(4)
    input_values = {}
    input_values['CoursePrice'] = row1[0].number_input('Course price ($)', min_value=0.0, max_value=2000.0, value=199.0, step=10.0, help='Set to 0 for a free course.')
    input_values['CourseDuration'] = row1[1].number_input('Duration (hours)', min_value=1.0, max_value=100.0, value=float(round(DATA['CourseDuration'].median(), 1)), step=1.0)
    input_values['CourseRating'] = row1[2].slider('Course rating', 1.0, 5.0, float(round(DATA['CourseRating'].mean(), 2)), 0.01)
    input_values['TeacherRating'] = row1[3].slider('Teacher rating', 1.0, 5.0, float(round(DATA['TeacherRating'].mean(), 2)), 0.01)
    row2 = st.columns(5)
    input_values['CourseLevel'] = row2[0].selectbox('Course level', safe_options(DATA['CourseLevel']), index=min(1, DATA['CourseLevel'].nunique()-1))
    input_values['CourseCategory'] = row2[1].selectbox('Category', safe_options(DATA['CourseCategory']))
    input_values['CourseType'] = row2[2].selectbox('Course type', safe_options(DATA['CourseType']))
    input_values['YearsOfExperience'] = row2[3].number_input('Teacher experience', min_value=0, max_value=50, value=int(round(DATA['YearsOfExperience'].median())), step=1)
    input_values['Expertise'] = row2[4].selectbox('Teacher expertise', safe_options(DATA['Expertise']))

live_features = add_live_features(input_values, DATA)
predictions = {target_name: float(target_info['best_model'].predict(live_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0]) for target_name, target_info in MODELS.items()}

st.markdown('### Prediction results')
results = st.columns(5)
results[0].metric('Predicted enrollment', f"{max(0, predictions['Enrollment Count']):,.0f}")
results[1].metric('Predicted course revenue', money(max(0, predictions['Course Revenue'])))
results[2].metric('Predicted category revenue', money(max(0, predictions['Category Revenue'])))
results[3].metric('Best performing model', MODELS['Enrollment Count']['best_name'])
r2 = MODELS['Enrollment Count']['table'].iloc[0]['R2 Score']
results[4].metric('Model confidence', f"{np.clip((r2 + 1) / 2, 0, 1):.0%}")

st.markdown('### Scenario feature profile')
st.dataframe(live_features, use_container_width=True, hide_index=True)
