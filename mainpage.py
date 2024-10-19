import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import numpy as np
import streamlit_antd_components as sac



st.set_page_config(
    page_title="MESS - Wastage Predictor",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded")
st.markdown(" <style> div[class^='st-emotion-cache-1gwvy71'] { padding: 0rem; } </style> ", unsafe_allow_html=True)
alt.themes.enable("dark")
st.sidebar.container(height=100).markdown("<h1>MESS - Wastage Predictor</h1>", unsafe_allow_html=True)
st.markdown('''
        <style>
        .fullHeight {
            height: 85vh;
            width: 95%;
            padding: 0;
        }
        </style>''', unsafe_allow_html=True)
    
#     container = st.container()
#     container.markdown("<div scr='linke', class = 'fullHeight'></div>", unsafe_allow_html = True)
    # st.title('🍫 MESS - Wastage Predictor",')
    # sac.segmented(
    #     items=[
    #         sac.SegmentedItem(label='A Mess'),
    #         sac.SegmentedItem(label ='C Mess'),
    #         sac.SegmentedItem(label='D Mess'),
    #     ], label='Select Mess', align='center', divider= False
    # )
    # sac.buttons([
    # sac.ButtonsItem(label='button'),
    # sac.ButtonsItem(icon='apple'),
    # sac.ButtonsItem(label='google', icon='google'),
    # sac.ButtonsItem(label='wechat', icon='wechat'),
    # # Add custom CSS to make the sidebar height 100% of the screen height

    # sac.ButtonsItem(label='link', icon='share-fill', href='https://ant.design/components/button'),]
    # , label='Others', align='center')

    # st.markdown("---")
    # st.write("Name: John Doe")
    # st.write("Email: john.doe@example.com")
    # sac.buttons([
    #     sac.ButtonsItem(icon='power', label='Log Out')
    # ], align='center')
 
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        ) 
    # height=300
    return heatmap

# def make_choropleth(input_df, input_id, input_column, input_color_theme):
#     choropleth = px.choropleth(input_df, locations=input_id, color=input_column, locationmode="USA-states",
#                                color_continuous_scale=input_color_theme,
#                                range_color=(0, max(df_selected_year.population)),
#                                scope="usa",
#                                labels={'population':'Population'}
#                               )
#     choropleth.update_layout(
#         template='plotly_dark',
#         plot_bgcolor='rgba(0, 0, 0, 0)',
#         paper_bgcolor='rgba(0, 0, 0, 0)',
#         margin=dict(l=0, r=0, t=0, b=0),
#         height=350
#     )
#     return choropleth

def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text

def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'

col = st.columns((1.5, 4.5, 2), gap='medium')

with col[1]:
    st.markdown('#### BITS Goa')
    st.image(f'map.png')
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    st.markdown('#### Graph') 
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    st.line_chart(chart_data)
    with st.container():
        st.markdown("### Expected Footfall")
        migrations_col = st.columns(3)
        with migrations_col[0]:
            Expected_footfall_B = 62
            donut_chart_Expected_footfall_b = make_donut(Expected_footfall_B, 'Expected Footfall', 'green')
            st.write('Breakfast')
            st.altair_chart(donut_chart_Expected_footfall_b)
        with migrations_col[1]:
            Expected_footfall_L = 80
            donut_chart_Expected_footfall_l = make_donut(Expected_footfall_L, 'Expected Footfall', 'green')
            st.write('Lunch')
            st.altair_chart(donut_chart_Expected_footfall_l)
        with migrations_col[2]:
            Expected_footfall_D = 90
            donut_chart_Expected_footfall_d = make_donut(Expected_footfall_D, 'Expected Footfall', 'green')
            st.write('Dinner')
            st.altair_chart(donut_chart_Expected_footfall_d)
    # heatmap = make_heatmap(df_reshaped, 'year', 'states', 'population', selected_color_theme)
    # st.altair_chart(heatmap, use_container_width=True)


    
    with st.expander('About', expanded=True):
        st.write('''
            - Data: [U.S. Census Bureau](<https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html>).
            - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
            - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
            ''')
        
