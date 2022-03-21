import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Data
df = px.data.gapminder()
viz_df = pd.DataFrame([[41, 20], [46.227638, 2.213749]], columns=["lat", "lon"])

# Header of Streamlit app
st.header("Agricultural Crop Yield Analysis")

# Sidebar to switch pages
page = st.sidebar.selectbox(
    "Select page", ["Interactive Map", "Per Country", "Per Continent"]
)

# Countries data
if page == "Per Country":
    ## Countries
    clist = df["country"].unique()
    country = st.selectbox("Select a country:", clist)
    col1, col2 = st.columns(2)
    fig = px.line(
        df[df["country"] == country], x="year", y="gdpPercap", title="GDP per Capita"
    )
    col1.plotly_chart(fig, use_container_width=True)
    fig = px.line(
        df[df["country"] == country], x="year", y="pop", title="Population Growth"
    )
    col2.plotly_chart(fig, use_container_width=True)

# Interactive Map
elif page == "Interactive Map":
    fig1 = px.scatter_geo(
        df,
        locations="country",
        locationmode="country names",
        color="continent",
        hover_name="country",
        size="pop",
        animation_frame="year",
        projection="natural earth",
        size_max=50,
    )
    st.plotly_chart(fig1)

else:
    ## Continents
    contlist = df["continent"].unique()

    continent = st.selectbox("Select a continent:", contlist)
    col1, col2 = st.columns(2)
    fig = px.line(
        df[df["continent"] == continent],
        x="year",
        y="gdpPercap",
        title="GDP per Capita",
        color="country",
    )
    col1.plotly_chart(fig)
    fig = px.line(
        df[df["continent"] == continent],
        x="year",
        y="pop",
        title="Population",
        color="country",
    )
    col2.plotly_chart(fig, use_container_width=True)
