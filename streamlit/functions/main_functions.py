from datetime import datetime

# for data and saves
import pandas as pd

# for app
import streamlit as st

# for plots
import plotly.express as px


# **********************************************************
# *                      functions                         *
# **********************************************************

###
### AGRICULTURAL & CROPS DATA ###
###

# To Plot different graphs for Agricultural & Crops datasets
def show_data(df, continent, area, item, element, year=0):
    year = "Y" + str(year)

    # ALL countries, 1 item, ALL element => 3 barplots ('Total Area Harvested, Yield & production of ITEM per country per YEAR with slider')
    if (area == "ALL") & (element == "ALL"):
        for el in df["Element"].unique():
            to_plot = df[
                (df["Continent"] == continent)
                & (df["Element"] == el)
                & (df["Item"] == item)
            ][[year, "Area"]]
            fig = px.bar(
                to_plot,
                x="Area",
                y=year,
                labels={
                    year: "Total "
                    + str(el)
                    + " ("
                    + str(df[(df["Element"] == el)]["Unit"].unique()[0])
                    + ")",
                    "Area": "Countries",
                },
                title="Total "
                + str(el)
                + " of "
                + str(item)
                + " per country in "
                + str(continent)
                + " ("
                + str(year[1:])
                + ")",
                width=1000,
                height=500,
                template="plotly",
            )

            st.plotly_chart(fig)

    # ALL countries, 1 item, 1 element => 1 barplot ('ELEMENT of ITEM per country per YEAR with slider')
    elif (area == "ALL") & (element != "ALL"):
        to_plot = df[
            (df["Continent"] == continent)
            & (df["Element"] == element)
            & (df["Item"] == item)
        ][[year, "Area"]]
        fig = px.bar(
            to_plot,
            x="Area",
            y=year,
            labels={
                year: "Total "
                + str(element)
                + " "
                + df[df["Element"] == element]["Unit"].unique()[0],
                "Area": "Countries",
            },
            title="Total "
            + str(element)
            + " of "
            + str(item)
            + " per country in "
            + str(continent)
            + " ("
            + str(year[1:])
            + ")",
            width=1000,
            height=500,
            template="plotly",
        )
        st.plotly_chart(fig)

    # 1 country, 1 item, ALL elements => 3 lineplots ('Area harvested', 'Yield','Production') per YEAR (linegrah, no slider)
    elif area != "ALL":
        if element == "ALL":
            for el in df["Element"].unique():
                tmp_df = df[
                    (df["Continent"] == continent)
                    & (df["Element"] == el)
                    & (df["Area"] == area)
                    & (df["Item"] == item)
                ].drop(
                    columns=[
                        "Continent",
                        "Area Code",
                        "Area",
                        "Item Code",
                        "Item",
                        "Element Code",
                        "Element",
                        "Unit",
                    ]
                )
                to_plot = tmp_df.T.set_axis(["Value"], axis=1, inplace=False)
                to_plot.index = to_plot.index.map(lambda x: x[1:])
                to_plot.index = to_plot.index.map(lambda x: "01/01/" + str(x))
                to_plot.index = pd.to_datetime(to_plot.index)
                to_plot.index.names = ["Year"]
                fig = px.line(
                    to_plot,
                    x=to_plot.index,
                    y="Value",
                    labels={
                        "Value": "Total "
                        + str(el)
                        + " of "
                        + str(item)
                        + " ("
                        + str(df[(df["Element"] == el)]["Unit"].unique()[0])
                        + ")"
                    },
                    title="Evolution of the Total "
                    + str(el)
                    + " of "
                    + str(item)
                    + " in "
                    + str(area),
                    width=1000,
                    height=500,
                    template="plotly",
                )
                st.plotly_chart(fig)
        else:
            tmp_df = df[
                (df["Continent"] == continent)
                & (df["Element"] == element)
                & (df["Area"] == area)
                & (df["Item"] == item)
            ].drop(
                columns=[
                    "Continent",
                    "Area Code",
                    "Area",
                    "Item Code",
                    "Item",
                    "Element Code",
                    "Element",
                    "Unit",
                ]
            )
            to_plot = tmp_df.T.set_axis(["Value"], axis=1, inplace=False)
            to_plot.index = to_plot.index.map(lambda x: x[1:])
            to_plot.index = to_plot.index.map(lambda x: "01/01/" + str(x))
            to_plot.index = pd.to_datetime(to_plot.index)
            to_plot.index.names = ["Year"]
            fig = px.line(
                to_plot,
                x=to_plot.index,
                y="Value",
                labels={
                    "Value": "Total "
                    + str(element)
                    + " of "
                    + str(item)
                    + " ("
                    + str(df[(df["Element"] == element)]["Unit"].unique()[0])
                    + ")"
                },
                title="Evolution of the Total "
                + str(element)
                + " of "
                + str(item)
                + " in "
                + str(area),
                width=1000,
                height=500,
                template="plotly",
            )
            st.plotly_chart(fig)


# Interactive Map function
def show_interactive_map(df, item, element):
    to_plot = df[(df["Item"] == item) & (df["Element"] == element)].drop(
        columns=[
            "Area Code",
            "Item Code",
            "Item",
            "Element Code",
            "Element",
            "Unit",
        ]
    )
    to_plot = to_plot.melt(
        id_vars=["Continent", "Area"], var_name="Year", value_name="Value"
    )
    to_plot["Year"] = to_plot["Year"].apply(lambda x: x[1:])
    to_plot["Value"] = to_plot["Value"].fillna(0)
    fig = px.scatter_geo(
        to_plot,
        locations="Area",
        locationmode="country names",  # ‘ISO-3’, ‘USA-states’, or ‘country names’
        color="Continent",  # Values from this column or array_like are used to assign color to marks
        hover_name="Area",
        size="Value",  # size of circles
        animation_frame="Year",
        projection="natural earth",
        size_max=30,
        width=1000,
        height=580,
        template="plotly",
        title=f"Evolution of the total {element} of {item}",
    )
    st.plotly_chart(fig)


def show_descriptive_scatter(df, item):
    # FIRST, since we do not have the same number of data for the different elements, let's select only the items where we have all the 3 infos
    items_to_drop = list(
        set(df["Item"].unique()) - set(df[(df["Element"] == "Yield")]["Item"].unique())
    )
    to_plot = df[~df["Item"].isin(items_to_drop)]

    # Let's now drop the columns we do not need
    to_plot = to_plot[(to_plot["Item"] == item)].drop(
        columns=[
            "Area Code",
            "Item Code",
            "Item",
            "Element Code",
            "Unit",
        ]
    )
    # let's deflate our df to put all our "Year" columns in a single column
    to_plot = to_plot.melt(
        id_vars=["Continent", "Area", "Element"], var_name="Year", value_name="Value"
    )

    # Drop countries without all info because not all countries harvest or have this info for THIS particular item
    countries_to_drop = list(
        set(to_plot["Area"].unique())
        - set(to_plot[(to_plot["Element"] == "Yield")]["Area"].unique())
    )
    to_plot = to_plot[~to_plot["Area"].isin(countries_to_drop)]

    # Some small slicing to get the Year
    to_plot["Year"] = to_plot["Year"].apply(lambda x: x[1:])

    # Fill NaN values with 0 to be able to plot
    to_plot["Value"] = to_plot["Value"].fillna(0)

    # Define our variables for our plot
    Area_harvested = to_plot[to_plot["Element"] == "Area harvested"]["Value"]
    Production = to_plot[to_plot["Element"] == "Production"]["Value"]
    size = to_plot[to_plot["Element"] == "Yield"]["Value"]
    countries = to_plot[to_plot["Element"] == "Production"]["Area"]
    years = to_plot[to_plot["Element"] == "Production"]["Year"]
    continents = to_plot[to_plot["Element"] == "Production"]["Continent"]

    # Plot our scatter plot
    fig = px.scatter(
        to_plot,
        x=Area_harvested,
        y=Production,
        labels={
            "x": "Area Harvested",
            "y": "Production",
        },
        animation_frame=years,
        animation_group=countries,
        size=size,
        color=continents,
        hover_name=countries,
        facet_col=continents,
        log_x=True,
        size_max=30,
        range_x=[1, max(Area_harvested)],
        range_y=[1, max(Production)],
        width=1000,
        height=580,
        template="plotly",
        title=f"Descriptive evolution ({item})",
    )
    st.plotly_chart(fig)


###
### GAS EMISSIONS DATA ###
###

###
### POPULATION DATA ###
###


# To get year to year list
def daterange(start_dt, end_dt):
    for n in range(int(end_dt.year) - int(start_dt.year) + 1):
        yield int(start_dt.year) + n
