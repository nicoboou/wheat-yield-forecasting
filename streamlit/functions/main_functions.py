from datetime import datetime
from turtle import width

# for data and saves
import pandas as pd
import numpy as np

# for app
import streamlit as st

# for plots
import plotly.express as px
from matplotlib import pyplot as plt

from statsmodels.tsa.stattools import adfuller
import math
from sklearn.model_selection import train_test_split


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


# Display choropleth map for ITEM
def show_choropleth_map(df, item, element, year):
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
    to_plot = to_plot[to_plot["Year"] == str(year)]
    fig = px.choropleth(
        to_plot,
        locations="Area",
        locationmode="country names",
        color="Value",  #  It is a COLUMN of df
        hover_name="Area",  # Another COLUMN of df
        color_continuous_scale=px.colors.sequential.Greens,
        title=f"{element} of {item} in {year}",
        width=1000,
        height=580,
    )
    st.plotly_chart(fig)


# Display imgs
def display_img(index):
    empty1, col, empty2 = st.columns([2.5, 20, 2.5])
    img = st.session_state.imgs[index]
    col.image(img, use_column_width="always")
    return


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
        size_max=45,
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
        size_max=35,
        range_x=[1, max(Area_harvested)],
        range_y=[1, max(Production)],
        width=1000,
        height=580,
        template="plotly",
        title=f"Descriptive evolution ({item})",
    )
    st.plotly_chart(fig)


def show_most_productive_crops(df, year):
    tmp_df = df.drop(columns=["Area Code", "Item Code", "Element Code", "Unit"])
    tmp_df = tmp_df.melt(
        id_vars=["Continent", "Area", "Item", "Element"],
        var_name="Year",
        value_name="Total Worldwide Yield (hg/ha)",
    )
    # to display the rank dataframe
    tmp_df["Year"] = tmp_df["Year"].apply(lambda x: x[1:])
    tmp_df = tmp_df[(tmp_df["Year"] == str(year)) & (tmp_df["Element"] == "Yield")]

    total_yield = pd.pivot_table(
        tmp_df, values="Total Worldwide Yield (hg/ha)", columns=["Item"], aggfunc=np.sum
    ).T
    total_yield = total_yield.reset_index()
    total_yield = (
        total_yield.sort_values(by="Total Worldwide Yield (hg/ha)", ascending=False)
        .reset_index()
        .drop(columns=["index"])
        .rename_axis("RANK")
        .reset_index()
    )

    st.dataframe(total_yield)

    # Productivity ranking of Wheat & Barley
    st.subheader(f"Productivity Ranking of Wheat & Barley in {year}")
    feat0, val0 = st.columns([3.5, 3.5])
    feat1, val1 = st.columns([3.5, 3.5])
    feat0.warning("Wheat:")
    val0.info(total_yield[total_yield["Item"] == "Wheat"].index[0])
    feat0.warning("Barley:")
    val0.info(total_yield[total_yield["Item"] == "Barley"].index[0])


def show_productivity_evolution(df):
    tmp_df = df.drop(columns=["Area Code", "Item Code", "Element Code", "Unit"])
    tmp_df = tmp_df.melt(
        id_vars=["Continent", "Area", "Item", "Element"],
        var_name="Year",
        value_name="Value",
    )
    tmp_df = tmp_df[(tmp_df["Element"] == "Yield")]
    total_yield = pd.pivot_table(
        tmp_df, values="Value", index=["Item"], columns=["Year"], aggfunc=np.sum
    ).T

    top2_to_plot = total_yield.loc[:, ["Wheat", "Barley"]]
    top5_to_plot = total_yield.loc[
        :,
        [
            "Mushrooms and truffles",
            "Sugar Crops Primary",
            "Tomatoes",
            "Wheat",
            "Barley",
        ],
    ]
    fig, axs = plt.subplots(2, figsize=(13, 13), dpi=120)
    axs[0].plot(top2_to_plot)
    axs[0].legend(["Wheat", "Barley"], loc="upper right")
    axs[0].set_title("Evolution of yield (Wheat, Barley)")
    axs[1].plot(top5_to_plot)
    axs[1].legend(
        [
            "Mushrooms and truffles",
            "Sugar Crops Primary",
            "Tomatoes",
            "Wheat",
            "Barley",
        ],
        loc="upper right",
    )
    axs[1].set_title("Comparison of productivity over the years")
    st.pyplot(fig)


def show_most_productive_countries(df, year):
    # to display the piechart
    item = st.selectbox(
        label="Select a food type", options=["-"] + list(df["Item"].unique())
    )
    tmp_df = df.drop(columns=["Area Code", "Item Code", "Element Code", "Unit"])
    tmp_df = tmp_df.melt(
        id_vars=["Continent", "Area", "Item", "Element"],
        var_name="Year",
        value_name="Value",
    )
    tmp_df["Year"] = tmp_df["Year"].apply(lambda x: x[1:])
    tmp_df = tmp_df[
        (tmp_df["Year"] == str(year))
        & (tmp_df["Element"] == "Production")
        & (tmp_df["Item"] == item)
    ]
    tmp_df = tmp_df.sort_values(by="Value", ascending=False).reset_index().fillna(0)
    tmp_df.loc[
        tmp_df["Value"] < (0.03 * tmp_df["Value"].max()),
        "Area",
    ] = "Other countries"  # Represent only large countries
    fig = px.pie(
        tmp_df,
        values="Value",
        names="Area",
        title=f"Share of production of {item} per country ({year})",
        width=1000,
        height=800,
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


### MACHINE LEARNING


def stationarity_test(df, column):

    # Size of plot
    fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)

    # Determine rolling statistics
    movingAverage = df[column].rolling(window=10).mean()
    movingSTD = df[column].rolling(window=10).std()

    # Plot rolling statistics
    orig = plt.plot(df[column], color="blue", label="Original")
    mean = plt.plot(movingAverage, color="red", label="Rolling Mean")
    std = plt.plot(movingSTD, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard Deviation")
    st.pyplot(fig)

    # Augmented Dickey–Fuller test:
    st.markdown("_Results of Augmented Dickey Fuller test:_")
    dick_full_test = adfuller(df[column], autolag="AIC")
    dick_full_output = pd.Series(
        dick_full_test[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dick_full_test[4].items():
        dick_full_output["Critical Value (%s)" % key] = value

    st.dataframe(dick_full_output.to_frame("Results"))


def from_pred_to_val(raw_predictions, logscaled_history):

    """ Initial datapoint: the datapoint where you want to initiate your cumulative sum"""

    # raw_predictions need to be model_results.fittedvalues
    predictions_ARIMA_diff = pd.Series(raw_predictions, copy=True)

    # Convert to cumulative sum
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

    # logscaled_history = df_logscaled['Total_Wheat_Prod']
    predictions_ARIMA_log = pd.Series(
        logscaled_history.iloc[0], index=predictions_ARIMA_diff_cumsum.index
    )

    predictions_ARIMA_log = predictions_ARIMA_log.add(
        predictions_ARIMA_diff_cumsum, fill_value=0
    )

    # Inverse of log is exp.
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    return predictions_ARIMA


def from_series_to_supervised(df):
    # Import our df
    wheat_prod = df["Total_Wheat_Prod"].values

    # We create a new df by using the max_length preceding years as features for our target_year
    full_years = []
    step = 1
    max_len = 5  # Here we are using the XX preceding years

    for idx in range(0, len(wheat_prod) - max_len, step):
        full_years.append(wheat_prod[idx : idx + max_len + 1])

    columns_names = [f"Value_Year -{max_len - idx}" for idx in range(0, max_len)]
    columns_names.append("Target_Year")

    expanded_wheat_prod = pd.DataFrame(
        full_years, columns=columns_names, index=df.index[max_len:]
    )
    # Train & test sets created
    X = expanded_wheat_prod.iloc[:, :-1]
    y = expanded_wheat_prod.iloc[:, -1]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    st.dataframe(expanded_wheat_prod.tail())
    st.markdown("*Shape of train & test dataset*")
    st.write(X_train.shape, X_valid.shape)
    st.write(y_train.shape, y_valid.shape)
    return expanded_wheat_prod, X, y, X_train, X_valid, y_train, y_valid


def print_score(m, X_train, y_train, X_valid, y_valid):
    st.write("RMSE on train set: {:.4f}".format(rmse(m.predict(X_train), y_train)))
    st.write("RMSE on valid set: {:.4f}".format(rmse(m.predict(X_valid), y_valid)))
    st.write("R^2 on train set: {:.4f}".format(m.score(X_train, y_train)))
    st.write("R^2 on valid set: {:.4f}".format(m.score(X_valid, y_valid)))
    if hasattr(m, "oob_score_"):
        st.write("R^2 on oob set: {:.4f}".format(m.oob_score_))
    return


def rmse(y_gold, y_pred):
    return math.sqrt(((y_gold - y_pred) ** 2).mean())


def rf_forecast(model, df, steps_to_forecast=10):
    new_df = df.copy()
    for i in range(1, steps_to_forecast):
        previous_years = new_df.iloc[len(new_df.index) - 1, 1:].values
        predicted_year = model.predict([previous_years])
        full_years = np.append(previous_years, predicted_year)
        new_df.loc[new_df.index.shift(1, freq="AS")[-1]] = full_years
    return new_df


def features_importances(df, estimator):
    listed_importance = {}
    for index, importance in enumerate(estimator.feature_importances_):
        feature_name = df.columns[index]
        listed_importance[feature_name] = importance
    sorted_importance = dict(
        sorted(listed_importance.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_importance


def plot_forecast_scenarios(
    df, temperature_scenario, precipitations_scenario, pop_scenario
):
    to_plot = df[
        (df["Precipitations Type"] == precipitations_scenario)
        & (df["Temperature Type"] == temperature_scenario)
        & (df["Variant"] == pop_scenario)
    ]
    plt.plot(
        to_plot.index,
        to_plot["Total_Wheat_YIELD"],
        ls="--",
        lw="2",
        label=f"Forecasted ({temperature_scenario}/{precipitations_scenario}/{pop_scenario})",
    )
