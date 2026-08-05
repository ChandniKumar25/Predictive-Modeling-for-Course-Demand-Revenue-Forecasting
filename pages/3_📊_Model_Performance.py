import streamlit as st
import plotly.express as px
from common import MODELS, TARGETS, chart_layout, init_page

init_page('03 · Model performance', 'Validation leaderboard')
performance_target = st.selectbox('Choose a target', list(TARGETS.keys()))
performance_table = MODELS[performance_target]['table'].copy()
left, right = st.columns([1.2, .8])
with left:
    fig = px.bar(performance_table, x='Model', y='RMSE', color='R2 Score', title=f'{performance_target} · validation RMSE', color_continuous_scale=['#2451b8','#4dd9ff'])
    st.plotly_chart(chart_layout(fig, 430), use_container_width=True)
with right:
    st.dataframe(performance_table.style.format({'MAE':'{:.2f}', 'RMSE':'{:.2f}', 'R2 Score':'{:.3f}'}), use_container_width=True, hide_index=True)
    st.success(f"Deployment choice · {MODELS[performance_target]['best_name']}")

st.markdown('### Model selection summary')
summary_rows = []
for target_name, target_info in MODELS.items():
    best_row = target_info['table'].iloc[0]
    summary_rows.append({'Target': target_name, 'Selected model': target_info['best_name'], 'MAE': best_row['MAE'], 'RMSE': best_row['RMSE'], 'R² Score': best_row['R2 Score']})
summary = __import__('pandas').DataFrame(summary_rows)
st.dataframe(summary.style.format({'MAE':'{:.2f}', 'RMSE':'{:.2f}', 'R² Score':'{:.3f}'}), use_container_width=True, hide_index=True)
