import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from common import DATA, chart_layout, init_page

init_page('02 · EDA', 'Exploratory Data Analysis')

st.markdown('### Dataset overview')
metric_cols = st.columns(5)
metric_cols[0].metric('Rows', f'{len(DATA):,}')
metric_cols[1].metric('Columns', f'{DATA.shape[1]:,}')
metric_cols[2].metric('Courses', f'{DATA.CourseID.nunique():,}')
metric_cols[3].metric('Teachers', f'{DATA.TeacherID.nunique():,}')
metric_cols[4].metric('Missing cells', f'{int(DATA.isna().sum().sum()):,}')

eda_tabs = st.tabs(['Distributions', 'Category & instructor insights', 'Correlation analysis'])
with eda_tabs[0]:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(DATA, x='Amount', nbins=40, title='Transaction amount distribution', color_discrete_sequence=['#4dd9ff'])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with right:
        fig = px.histogram(DATA, x='CourseRating', nbins=20, title='Course rating distribution', color='CourseLevel', marginal='box')
        st.plotly_chart(chart_layout(fig), use_container_width=True)
with eda_tabs[1]:
    category = DATA.groupby('CourseCategory', as_index=False).agg(Enrollments=('TransactionID','count'), Revenue=('Amount','sum'), AvgRating=('CourseRating','mean')).sort_values('Revenue', ascending=False)
    left, right = st.columns(2)
    with left:
        fig = px.bar(category, x='CourseCategory', y='Enrollments', title='Enrollments by category', color='Enrollments', color_continuous_scale=['#2451b8','#4dd9ff'])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with right:
        instructor = DATA.groupby('TeacherName', as_index=False).agg(Enrollments=('TransactionID','count'), Revenue=('Amount','sum'), TeacherRating=('TeacherRating','mean')).sort_values('Revenue', ascending=False).head(15)
        fig = px.bar(instructor.sort_values('Revenue'), x='Revenue', y='TeacherName', orientation='h', title='Top instructors by revenue', color='TeacherRating', color_continuous_scale=['#2451b8','#4dd9ff'])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.dataframe(category.style.format({'Revenue':'${:,.0f}', 'AvgRating':'{:.2f}'}), use_container_width=True, hide_index=True)
with eda_tabs[2]:
    numeric = DATA.select_dtypes(include='number')
    corr = numeric.corr().round(2)
    fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Blues', showscale=True)
    st.plotly_chart(chart_layout(fig, 600), use_container_width=True)
