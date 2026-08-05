import streamlit as st
import plotly.express as px
from common import DATA, chart_layout, init_page, money

init_page()

st.markdown("""
<div class="hero">
<div class="smallcaps">Enterprise learning intelligence</div>
<h1>Predictive Modeling for Course Demand & Revenue Forecasting</h1>
<p>Make sharper portfolio, pricing, and instructor decisions with a live machine-learning workspace built on the EduPro platform data.</p>
<span class="badge">📈 Enrollment Forecasting</span><span class="badge">💰 Revenue Forecasting</span><span class="badge">🧠 Predictive Analytics</span><span class="badge">⚙️ Machine Learning</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section"><div class="smallcaps">01 · Executive overview</div><h2>Platform pulse</h2></div>', unsafe_allow_html=True)
metric_cols = st.columns(5)
metric_cols[0].metric("Gross revenue", money(DATA["Amount"].sum()))
metric_cols[1].metric("Avg. transaction", money(DATA["Amount"].mean()))
metric_cols[2].metric("Avg. course rating", f"{DATA['CourseRating'].mean():.2f} / 5")
metric_cols[3].metric("Paid mix", f"{DATA['IsPaidCourse'].mean():.0%}")
metric_cols[4].metric("Category coverage", f"{DATA['CourseCategory'].nunique()} categories")

left, right = st.columns([1.15, 1])
with left:
    summary = DATA.groupby('CourseCategory', as_index=False).agg(Enrollments=('TransactionID','count'), Revenue=('Amount','sum')).sort_values('Revenue', ascending=False)
    fig = px.bar(summary, x='Revenue', y='CourseCategory', orientation='h', title='Revenue by category', color='Revenue', color_continuous_scale=['#2451b8','#4dd9ff'])
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(chart_layout(fig), use_container_width=True)
with right:
    level = DATA.groupby('CourseLevel', as_index=False).agg(Enrollments=('TransactionID','count'), Revenue=('Amount','sum'))
    fig = px.scatter(level, x='Enrollments', y='Revenue', size='Revenue', color='CourseLevel', title='Demand and revenue by course level', text='CourseLevel')
    fig.update_traces(textposition='top center')
    st.plotly_chart(chart_layout(fig), use_container_width=True)

st.markdown('<div class="section"><div class="smallcaps">Executive summary</div><h2>What the data says</h2></div>', unsafe_allow_html=True)
insight_cols = st.columns(3)
with insight_cols[0]:
    top_category = summary.iloc[0]['CourseCategory']
    st.markdown(f'<div class="insight"><b>Category leader</b><br>{top_category} is the largest revenue contributor in the current platform snapshot.</div>', unsafe_allow_html=True)
with insight_cols[1]:
    st.markdown(f'<div class="insight"><b>Monetization mix</b><br>{DATA["IsPaidCourse"].mean():.0%} of observed transactions are associated with paid courses, making price positioning a key lever.</div>', unsafe_allow_html=True)
with insight_cols[2]:
    st.markdown('<div class="insight"><b>Next best action</b><br>Use the Live Prediction Studio to test pricing, course level, category, and instructor quality before launch.</div>', unsafe_allow_html=True)
