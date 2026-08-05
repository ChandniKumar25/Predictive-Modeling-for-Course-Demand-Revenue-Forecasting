import streamlit as st
import plotly.express as px
from common import MODELS, TARGETS, chart_layout, feature_importance_table, init_page

init_page('05 · Explainability', 'Feature importance & demand drivers')
importance_target = st.selectbox('Feature importance target', list(TARGETS.keys()))
importance_table = feature_importance_table(MODELS[importance_target]['best_model'])
left, right = st.columns([1.2, .8])
with left:
    fig = px.bar(importance_table.sort_values('Importance'), x='Importance', y='Feature', orientation='h', title=f'{importance_target} · top drivers', color='Importance', color_continuous_scale=['#2451b8','#4dd9ff'])
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(chart_layout(fig, 470), use_container_width=True)
with right:
    st.dataframe(importance_table.style.format({'Importance':'{:.4f}'}), use_container_width=True, hide_index=True)

st.markdown('### Demand driver analysis')
for feature in importance_table.head(5)['Feature']:
    st.markdown(f'<div class="insight"><b>{feature}</b><br>This feature is among the strongest model signals for {importance_target.lower()}, so it should be included in launch planning and scenario testing.</div>', unsafe_allow_html=True)
