import streamlit as st
from common import DATA, init_page

init_page('07 · Data access', 'Dataset explorer')
metric_cols = st.columns(5)
metric_cols[0].metric('Rows', f'{len(DATA):,}')
metric_cols[1].metric('Columns', f'{DATA.shape[1]:,}')
metric_cols[2].metric('Numeric fields', f'{len(DATA.select_dtypes(include="number").columns):,}')
metric_cols[3].metric('Categorical fields', f'{len(DATA.select_dtypes(exclude="number").columns):,}')
metric_cols[4].metric('Missing cells', f'{int(DATA.isna().sum().sum()):,}')

st.markdown('### Preview of merged and cleaned data')
row_count = st.slider('Rows to preview', 10, 250, 50, 10)
st.dataframe(DATA.head(row_count), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown('### Dataset statistics')
    st.dataframe(DATA.describe(include='all').T, use_container_width=True)
with right:
    st.markdown('### Data information')
    info_table = __import__('pandas').DataFrame({'Column': DATA.columns, 'Type': DATA.dtypes.astype(str).values, 'Non-null': DATA.notna().sum().values, 'Unique': DATA.nunique().values})
    st.dataframe(info_table, use_container_width=True, hide_index=True)
