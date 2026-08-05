import streamlit as st
from common import DATA, MODELS, init_page, money

init_page('06 · Decision support', 'Business insights & recommendations')
summary = DATA.groupby('CourseCategory', as_index=False).agg(Enrollments=('TransactionID','count'), Revenue=('Amount','sum'), AvgRating=('CourseRating','mean')).sort_values('Revenue', ascending=False)
top_category = summary.iloc[0]['CourseCategory']

cols = st.columns(3)
with cols[0]:
    st.markdown(f'<div class="insight"><b>Pricing recommendation</b><br>Test a tiered paid offer for high-rating courses and use free courses as acquisition funnels into premium pathways.</div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown(f'<div class="insight"><b>Course launch recommendation</b><br>Prioritize {top_category} or categories with strong enrollment momentum, then validate the scenario in the prediction studio.</div>', unsafe_allow_html=True)
with cols[2]:
    st.markdown('<div class="insight"><b>Instructor recommendation</b><br>Pair high-rated instructors with category-aligned expertise and make experience visible in the course positioning.</div>', unsafe_allow_html=True)

st.markdown('### Strategic decision support')
recommendation_table = summary.copy()
recommendation_table['Revenue per enrollment'] = recommendation_table['Revenue'] / recommendation_table['Enrollments'].replace(0, 1)
st.dataframe(recommendation_table.style.format({'Revenue':'${:,.0f}', 'AvgRating':'{:.2f}', 'Revenue per enrollment':'${:,.2f}'}), use_container_width=True, hide_index=True)

st.markdown('### Deployment guardrails')
st.write('The application benchmarks five regressors for enrollment, course revenue, and category revenue. The model with the lowest validation RMSE is selected automatically, while MAE and R² remain visible for governance and comparison.')
