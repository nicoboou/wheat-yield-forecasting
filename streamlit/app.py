from multiprocessing.connection import wait
import os
import sys
import copy
from io import open
from datetime import datetime
from attr import field

# for data and saves
import pandas as pd
import numpy as np
import dill
from PIL import Image  # pillow package

# for Time-Series Statistical Analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# for app
import streamlit as st

# for plots
import plotly.express as px

# constants
from constants import *

# config
st.set_page_config(layout="wide")

# paths
path_to_repo = os.path.dirname(os.getcwd())
path_to_data = os.path.join(path_to_repo, "data")
path_to_img = os.path.join(path_to_repo, "img")


# custom package
sys.path.insert(0, os.path.join(path_to_repo, "utils"))
# from ..utils.utils import *
from functions.main_functions import *


# **********************************************************
#                     session state                         *
# **********************************************************

if "crops_all" not in st.session_state:
    # validation set given in notebook
    # n_valid = 12000

    ### LOAD DATASETS ###

    def open_all_datasets_df(item):
        df = pd.read_csv(
            "https://raw.githubusercontent.com/nicoboou/ml_eml/main/data/agriculture-crop-production/Production_Crops_E_"
            + str(item)
            + ".csv",
            low_memory=False,
            encoding="latin1",
        )
        df["Continent"] = item
        return df

    continents = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
    crops_raw = pd.DataFrame()
    for continent in continents:
        crops_raw = crops_raw.append(open_all_datasets_df(continent))
    crops_all = copy.deepcopy(crops_raw)
    crops_all = crops_all.loc[:, ~crops_all.columns.str.endswith("F")]
    crops_all = crops_all[crops_all.columns[~crops_all.columns.str.endswith("F")]]
    first_col = crops_all.pop("Continent")
    crops_all.insert(0, "Continent", first_col)

    flags_df = pd.read_csv(
        "https://raw.githubusercontent.com/nicoboou/ml_eml/main/data/agriculture-crop-production/flags.csv",
        encoding="latin1",
    )

    # 2. Load Climate USA dataset
    climate_usa = pd.read_excel(
        "https://raw.githubusercontent.com/nicoboou/ml_eml/main/data/temperature-%26-precipitations/climate_data_usa.xlsx",
        parse_dates=True,
    )
    climate_usa["Year"] = climate_usa["Year"].apply(lambda x: "01/01/" + str(x))
    climate_usa = climate_usa.set_index("Year")
    climate_usa.index = pd.to_datetime(climate_usa.index)

    # 3. Load Population dataset
    pop_df_raw = pd.read_csv(
        "https://raw.githubusercontent.com/nicoboou/ml_eml/main/data/pop_data_onu.csv",
        thousands=" ",
    )
    pop_df_raw["Time"] = pop_df_raw["Time"].apply(lambda x: "01/01/" + str(x))
    pop_df_raw = pop_df_raw.set_index("Time")
    pop_df_raw.index = pd.to_datetime(pop_df_raw.index)

    usa_pop = pop_df_raw[pop_df_raw["Location"] == "United States of America"]
    usa_pop_estimations = usa_pop["2020":]
    usa_pop_hist = usa_pop[:"2019"]

    ### LOAD MODELS ###

    # 1. ARIMA Model (UNIVARIATE)
    path_to_arima_model = os.path.join(os.getcwd(), "model", "arima.pickle")
    with open(path_to_arima_model, "rb") as file:
        arima_model = dill.load(file)

    # 2. Random Forest (UNIVARIATE)
    path_to_rf_uni = os.path.join(path_to_repo, "models", "RandomForest_Univariate.pkl")
    with open(path_to_rf_uni, "rb") as file:
        rf_uni = dill.load(file)

    # 3. Linear Regression (UNIVARIATE)
    path_to_lr_uni = os.path.join(
        path_to_repo, "models", "LinearRegression_Univariate.pkl"
    )
    with open(path_to_lr_uni, "rb") as file:
        lr_uni = dill.load(file)

    # 4. Random Forest (MULTIVARIATE)
    path_to_rf_multi = os.path.join(
        path_to_repo, "models", "RandomForest_Multivariate.pkl"
    )
    with open(path_to_rf_multi, "rb") as file:
        rf_multi = dill.load(file)

    # 5. Linear Regression (UNIVARIATE)
    path_to_lr_multi = os.path.join(
        path_to_repo, "models", "LinearRegression_Multivariate.pkl"
    )
    with open(path_to_lr_multi, "rb") as file:
        lr_multi = dill.load(file)

    ### LOAD IMAGES ###
    imgs = []
    img_files = os.listdir(path_to_img)
    for img_file in img_files:
        img = Image.open(os.path.join(path_to_img, img_file))
        img = np.array(img)
        imgs.append(img)

    ### STORE IN CACHE ###

    st.session_state.crops_all = crops_all
    st.session_state.flags_df = flags_df
    st.session_state.pop_df_raw = pop_df_raw
    st.session_state.usa_pop = usa_pop
    st.session_state.usa_pop_hist = usa_pop_hist
    st.session_state.usa_pop_estimations = usa_pop_estimations
    st.session_state.climate_usa = climate_usa
    st.session_state.imgs = imgs

    st.session_state.arima_model = arima_model
    st.session_state.rf_uni = rf_uni
    st.session_state.lr_uni = lr_uni
    st.session_state.rf_multi = rf_multi
    st.session_state.lr_multi = lr_multi

# **********************************************************
#                     main script                         *
# **********************************************************

### SIDEBAR CONFIGURATION ###

page = st.sidebar.selectbox(
    "Select page", ["Introduction", "Datasets", "Exploration", "Predictions"]
)

# INTRODUCTION
if page == "Introduction":
    st.header("Introduction")
    display_img(2)
    st.write(
        "For thousands of years, farming and animal husbandry have been at the heart of human concerns. It began 10,000 years ago with subsistence farming. Then, little by little, a production and trade agriculture was created."
    )
    st.write(
        "Today, information about markets and their organisation, techniques and know-how benefiting from advances in agronomy, high-tech products, instruments and methods developed by the agri-supply industries, are available to the farmer to obtain levels of production never before achieved in human history."
    )
    st.write(
        "However, the margins achieved by agricultural companies in developed countries remain highly variable, depending on fluctuating sales prices and on whether or not the States provide aid, while in other countries the situation of many farmers remains precarious. On the other hand, these recent industrial developments are leading some consumers in rich countries to question the quality of food, its safety and the consequences of modern methods on the environment."
    )
    st.markdown(
        "Even more worrying, **recent geopolitical events** and the **climate crisis** have shifted the main issue **from the quality** of food grown to the **_quantity available_**."
    )
    st.subheader("Goals")
    st.markdown(
        "- **Worldwide Wheat Production (Univariate)**: based *only* on past datapoints, try to project futur values"
    )
    st.markdown(
        "- **USA Wheat Yield (Multivariate)**: using **external features** (such as **population** data & **climate change** data), try to forecast future values of the time-series."
    )
    st.subheader("Three main points of interest")
    feat0, feat1, feat3 = st.columns([2.5, 2.5, 2.5])
    feat0.warning("Production")
    feat1.warning("Area harvested")
    feat3.warning("Yield")
    st.markdown(" ")
    st.markdown("**Production:** _quantity of food produced (tonnes)_ ")
    st.markdown("**Area harvested:** _area of food used for production (hectares)_")
    st.markdown("**Yield:** _Production/Area harvested (hg/ha)_")


# DATASETS
elif page == "Datasets":
    dataset = st.sidebar.selectbox(
        label="Select dataset",
        options=[
            "Agricultural & Crops",
            "Population",
            "USA Climate",
        ],
    )
    if dataset == "Agricultural & Crops":
        st.header("Agricultural & Crops Dataset")
        st.subheader("Summary")
        st.write(
            "Crop statistics for 173 products in Africa, America, Asia, Europe and Oceania, collected from 1961 to 2019."
        )
        st.subheader("Description")
        st.markdown(
            "Data from the Food and Agriculture Organization of the United Nations (FAO). Achieving food security for all is at the heart of FAO's efforts - ensuring that people have regular access to enough quality food to lead active and healthy lives. Our three main objectives are: the eradication of hunger, food insecurity and malnutrition; the eradication of poverty and the promotion of economic and social progress for all; and the sustainable management and use of natural resources, including land, water, air, climate and genetic resources, for the benefit of present and future generations. Primary crops, fibre crops. Crop statistics are recorded for 173 commodities, covering the following categories: Primary crops, Primary fibre crops, Cereals, Secondary cereals, Citrus, Fruit, Jute and related fibres, Oilcake equivalent, Primary oilseeds, Dry vegetables, Roots and tubers, Green fruits and vegetables and Melons. Data are expressed in terms of area harvested, quantity produced, yield and quantity of seed. The aim is to provide comprehensive coverage of production of all primary crops for all countries and regions of the world."
        )
        st.subheader("Main Sample")
        st.dataframe(st.session_state.crops_all)
        st.subheader("Link")
        st.write(
            "Source: Organisation des Nations Unies pour l'alimentation et l'agriculture (FAO)"
        )
        st.write(
            "https://github.com/nicoboou/ml_eml/tree/main/data/agriculture-crop-production"
        )

    elif dataset == "USA Climate":
        st.header("USA Climate Dataset")
        st.subheader("Summary")
        st.write(
            "This dataset was consolidated from the WorldBank data source, which provides data regarding Essential Climate Variables (namely here Precipitations and Temperature."
        )
        st.subheader("Description")
        st.markdown(
            "The World  Bank has both consolidated historical data and also gathered the main projections regarding precipitations and temperature in any country. I chose to focus on the USA since it is one of the main producers of cereals (see Exploration of Agricultural Dataset for more details!) "
        )
        st.subheader("Main Sample")
        st.dataframe(st.session_state.climate_usa)
        st.subheader("Link")
        st.write(
            "Source: Organisation des Nations Unies pour l'alimentation et l'agriculture (FAO)"
        )
        st.write("https://climateknowledgeportal.worldbank.org/download-data")
        st.write(
            "https://github.com/nicoboou/ml_eml/blob/main/data/temperature-%26-precipitations/climate_data_usa.xlsx"
        )

    elif dataset == "Population":
        st.header("Population Dataset")
        st.subheader("Description")
        st.markdown(
            "The FAOSTAT Population module contains time series data on population, by sex and urban/rural. The series consist of both estimates and projections for different periods as available from the original sources, namely:"
        )
        st.write(
            "1. Population data refers to the World Population Prospects: The 2019 Revision from the UN Population Division."
        )
        st.write(
            "2. Urban/rural population data refers to the World Urbanization Prospects: The 2018 Revision from the UN Population Division."
        )
        st.subheader("Main Sample")
        st.dataframe(st.session_state.pop_df_raw)
        # Explanation
        st.markdown(
            "When it boils down to forecasting population evolution, different scenarios are kept from experts."
        )
        st.markdown(
            "According to the INSEE, a _population projection_ gives a picture of what the future **size** and **structure** of the population by sex and age _might_ look like. It is based on knowledge of the _past trends_, and, for the future, on _assumptions_ made for three components: **fertility**, **mortality** and **migration**. Different evolution assumptions are made for each component, constituting **different _scenarios_**. The projections serve as a basis for long-term thinking, particularly in terms of collective development. They make it possible to analyse population trends if the assumptions are true, but are not forecasts. Individual behaviour, certain public policy actions, scientific progress or unforeseen events (weather events, epidemics) in the coming years may have a lasting effect and significantly influence trends, which the projections do not take into account."
        )
        #'Medium', 'High', 'Low', 'Constant fertility','Instant replacement', 'Zero migration', 'Constant mortality',
        # 'No change', 'Momentum', 'Median PI', 'Upper 80 PI', 'Lower 80 PI',
        # 'Upper 95 PI', 'Lower 95 PI'
        st.markdown("***Medium-variant projection***")
        feat0, val0 = st.columns([2.5, 5])
        feat0.warning("Medium-variant")
        val0.info(
            "_'Median of several thousand distinct trajectories of each demographic component derived using the probabilistic model of the variability in changes over time.'_"
        )
        st.markdown("***Fertility variants***")
        (
            feat1,
            val1,
        ) = st.columns([2.5, 5])
        feat1.warning("Low")
        val1.info(
            "_'Total fertility is projected to reach a fertility level that is 0.5 births above the total fertility in the medium variant.'_"
        )

        (
            feat3,
            val3,
        ) = st.columns([2.5, 5])
        feat3.warning("High")
        val3.info(
            "_'Total fertility is projected to reach a fertility level that is 0.5 births above the total fertility in the medium variant._'"
        )
        (
            feat4,
            val4,
        ) = st.columns([2.5, 5])
        feat4.warning("Constant fertility")
        val4.info(
            "_'Total fertility remains constant at the level estimated for 2015-2020.'_"
        )
        (
            feat5,
            val5,
        ) = st.columns([2.5, 5])
        feat5.warning("Instant replacement")
        val5.info(
            "_'Fertility for each country is set to the level necessary to ensure a net reproduction rate of 1.0 starting in 2020-2025.'_"
        )
        st.markdown("***Mortality variants***")
        (
            feat6,
            val6,
        ) = st.columns([2.5, 5])
        feat6.warning("Constant mortality")
        val6.info(
            "_'Uses the same fertility assumption (medium fertility), and international migration assumption as the medium variant.'_"
        )
        (
            feat7,
            val7,
        ) = st.columns([2.5, 5])
        feat7.warning("No change")
        val7.info("_'Both fertility and mortality are kept constant'_")
        st.markdown("***Migration variants***")
        (
            feat8,
            val8,
        ) = st.columns([2.5, 5])
        feat8.warning("Zero migration")
        val8.info(
            "_'Allows an assessment of the effect that non-zero net migration has on various population quantities'_"
        )
        (
            feat9,
            val9,
        ) = st.columns([2.5, 5])
        feat9.warning("No change")
        val9.info(
            "_'Has the same assumption about international migration as the medium variant but differs from the latter by having constant fertility and mortality'_"
        )
        st.markdown("***Momentum variant***")
        (
            feat10,
            val10,
        ) = st.columns([2.5, 5])
        feat10.warning("Momentum variant")
        val10.info(
            "Illustrates the impact of age structure on long-term population change(United Nations, 2017)"
        )
        st.write("_Credits to United Nations, 2019_")
        st.subheader("Link")
        st.write("Source: Organisation des Nations Unies")
        st.write("https://population.un.org/wpp/Download/Standard/Population/")
        st.write(
            "https://raw.githubusercontent.com/nicoboou/ml_eml/main/data/pop_data_onu.csv"
        )


# EXPLORATION
elif page == "Exploration":
    explore = st.sidebar.selectbox(
        label="Select dataset to explore",
        options=["Agricultural & Crops", "Population", "USA Climate"],
    )
    if explore == "Agricultural & Crops":
        # Selectors
        st.header("Agricultural & Crops")
        tmp_df = st.session_state.crops_all
        type_of_exploration = st.sidebar.selectbox(
            label="Select what you want to explore",
            options=["-"] + ["Overview", "Detailed Graphs", "Evolutive Map"],
        )
        if type_of_exploration == "-":
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h2 style='text-align: center; color: green;'> &#11013;&#65039 Choose what you want to explore! &#129517</h2>",
                unsafe_allow_html=True,
            )

        if type_of_exploration == "Overview":
            # - Carte montrant les pays producteurs de blé -vs- non producteurs
            year = st.slider(
                "Select a year",
                min_value=datetime(1961, 1, 1),
                max_value=datetime(2019, 1, 1),
                format="YYYY",
            ).year

            show_choropleth_map(st.session_state.crops_all, "Wheat", "Production", year)

            ## ON TWO DIFFERENT COLUMNS
            # - 85% de la production de blé provient de 10 pays:
            # - États Unis
            # - Russie
            # - Chine
            # - Union Européenne
            # => blé est au cœur d’enjeux stratégiques, géopolitiques et sanitaires

            # TOP 5 most productive types of crops per year
            ## Somme des rendements pour l'année 2019 pour tous les pays, pour les 165 variétés

            st.markdown(
                f"**What are the most productive types of crops in **{year}**? Wheat ? Barley ? Something else ? Let's see**"
            )
            show_most_productive_crops(st.session_state.crops_all, year)

            st.markdown(
                f"**In terms of _countries_, what are the ones who produce the most in {year} ?**"
            )
            show_most_productive_countries(st.session_state.crops_all, year)

            st.markdown(
                "**Let's see the evolution in the _'productivity'_ over the years**"
            )
            show_productivity_evolution(st.session_state.crops_all)

            st.subheader("CONCLUSION")
            st.markdown(
                "**It appears that the yield of Wheat and Barley haven't stopped increasing, however it is nothing compared to the yield of the most productive food crops.**"
            )

            # TOP 5 most productive countries per year

            # - Situation en 2020:
            # - France 1re exportateur européen: 13,4M de tonnes vendues à l’étranger
            # - Russie 1er exportateur mondiale
            # - Chine 1re producteur mondiale: 130M tonnes/an
            # - Russie + Ukraine + Kazakhstan: 20% des exportations mondiales

        if type_of_exploration == "Detailed Graphs":
            continent = st.selectbox(
                label="Select a continent",
                options=["-"] + list(tmp_df["Continent"].unique()),
            )
            area = st.selectbox(
                label="Select a country",
                options=["-"]
                + ["ALL"]
                + list(tmp_df[tmp_df["Continent"] == continent]["Area"].unique()),
            )
            if area != "ALL":
                item = st.selectbox(
                    label="Select a food type",
                    options=["-"]
                    + list(
                        tmp_df[
                            (tmp_df["Continent"] == continent)
                            & (tmp_df["Area"] == area)
                        ]["Item"].unique()
                    ),
                )

                element = st.selectbox(
                    label="Select an element",
                    options=["-"] + ["ALL"] + list(tmp_df["Element"].unique()),
                )

                year = 0

            else:
                item = st.selectbox(
                    label="Select a food type",
                    options=["-"]
                    + list(tmp_df[(tmp_df["Continent"] == continent)]["Item"].unique()),
                )

                element = st.selectbox(
                    label="Select an element",
                    options=["-"] + ["ALL"] + list(tmp_df["Element"].unique()),
                )

                year = st.slider(
                    "Year",
                    min_value=datetime(1961, 1, 1),
                    max_value=datetime(2019, 1, 1),
                    format="YYYY",
                ).year

            # Show data function
            if (
                (continent != str("-"))
                and (item != str("-"))
                and (area != str("-"))
                and (element != str("-"))
            ):
                show_data(
                    st.session_state.crops_all, continent, area, item, element, year
                )
            else:
                pass

        elif type_of_exploration == "Evolutive Map":
            # Selectors
            item = st.selectbox(
                label="Select a food type",
                options=["-"] + list(tmp_df["Item"].unique()),
            )

            element = st.selectbox(
                label="Select an element",
                options=["-"] + list(tmp_df["Element"].unique()),
            )

            if (item != str("-")) and (element != str("-")):
                # print_main_info()
                show_interactive_map(st.session_state.crops_all, item, element)
                show_descriptive_scatter(st.session_state.crops_all, item)
            else:
                pass

    elif explore == "USA Climate":
        st.header("USA Climate")
        st.markdown(
            "How do the different scenarios look like when plotted on a graph regarding **_Temperatures_** ?"
        )
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        axs.plot(st.session_state.climate_usa["Temperature (historic)"])
        axs.plot(
            st.session_state.climate_usa["Temperature (RCP2.6)"].loc["2020":],
            label="Temperature °C (Low emissions scenario)",
        )
        axs.plot(
            st.session_state.climate_usa["Temperature (RCP6)"].loc["2020":],
            label="Temperature °C (Medium emissions scenario)",
        )
        axs.plot(
            st.session_state.climate_usa["Temperature (RCP8.5)"].loc["2020":],
            label="Temperature °C (High emissions scenario)",
        )
        axs.legend()
        st.pyplot(fig)

        st.markdown("And for **_Precipitations_**?")
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        axs.plot(st.session_state.climate_usa["Precipitations (historic)"])
        axs.plot(
            st.session_state.climate_usa["Precipitations (RCP2.6)"].loc["2020":],
            label="Precipitations °C (RCP2.6)",
        )
        axs.plot(
            st.session_state.climate_usa["Precipitations (RCP6)"].loc["2020":],
            label="Precipitations °C (RCP6)",
        )
        axs.plot(
            st.session_state.climate_usa["Precipitations (RCP8.5)"].loc["2020":],
            label="Precipitations °C (RCP8.5)",
        )
        axs.legend()
        st.pyplot(fig)

    elif explore == "Population":
        st.header("Population")

        # Slider for Years selection
        year = st.slider(
            "Select a year",
            min_value=datetime(1950, 1, 1),
            max_value=datetime(2021, 1, 1),
            format="YYYY",
        ).year

        # Choropleth map

        to_plot2 = st.session_state.pop_df_raw[
            (st.session_state.pop_df_raw.index == str(year))
            & (st.session_state.pop_df_raw["Location"].isin(countries))
        ][["Location", "PopTotal", "PopMale", "PopFemale"]]
        to_plot2["PopTotal"] = to_plot2["PopTotal"].apply(lambda x: x * 1000)

        fig = px.choropleth(
            to_plot2,
            locations="Location",
            locationmode="country names",
            color="PopTotal",  #  It is a COLUMN of df
            hover_name="Location",  # Another COLUMN of df
            color_continuous_scale=px.colors.sequential.Redor,
            title=f"Repartition of WorldWide Population in {year} ",
            width=1000,
            height=580,
        )
        st.plotly_chart(fig)

        # Lineplots for WorldWide pop evolution

        to_plot1 = st.session_state.pop_df_raw[
            st.session_state.pop_df_raw["Location"] == "World"
        ][["PopMale", "PopFemale", "PopTotal", "Variant"]]
        fig = px.line(
            to_plot1,
            x=to_plot1.index,
            y="PopTotal",
            color="Variant",
            labels={"PopTotal": "Total WorldWide Population"},
            title="Evolution of the Total WorldWide Population (Historic & Forecast)",
            width=1030,
            height=900,
            template="plotly",
        )
        st.plotly_chart(fig)


# PREDICTIONS
elif page == "Predictions":
    prediction = st.sidebar.selectbox(
        label="Select forecasting experiment:",
        options=[
            "Worldwide Wheat Production (Univariate)",
            "USA Wheat Yield (Multivariate)",
        ],
    )
    if prediction == "Worldwide Wheat Production (Univariate)":
        st.markdown("## I. Time-series Analysis #1: Worldwide Wheat Production")

        st.markdown(
            "The first experiment I've tried to achieve is to predict the production of wheat (since it is the most valuable food currently) on a worldwide scale, using **_[univariate](#predictions)_** methods."
        )

        ### DATA PREPROCESSING
        st.markdown("### Data Preprocessing")
        st.markdown(
            "And first, in order to be able to make any use of our data in a time series analysis purpose, I will transform the datasets so that I have **dates** as index. It will be much easier to manipulate it as a time-series for forecasting."
        )

        wheat_prod_ts = st.session_state.crops_all[
            (st.session_state.crops_all["Item"] == "Wheat")
            & (st.session_state.crops_all["Element"] == "Production")
        ].drop(
            columns=[
                "Area Code",
                "Item Code",
                "Item",
                "Element Code",
                "Unit",
            ]
        )
        # let's deflate our df to put all our "Year" columns in a single column
        wheat_prod_ts = wheat_prod_ts.melt(
            id_vars=["Continent", "Area", "Element"],
            var_name="Year",
            value_name="Value",
        )

        # Some small slicing to get the Year
        wheat_prod_ts["Year"] = wheat_prod_ts["Year"].apply(lambda x: x[1:])

        # Fill NaN values with 0 to be able to plot
        wheat_prod_ts["Value"] = wheat_prod_ts["Value"].fillna(0)
        wheat_prod_ts = pd.pivot_table(
            wheat_prod_ts, values="Value", index=["Year"], aggfunc=np.sum
        )
        wheat_prod_ts.index = pd.to_datetime(wheat_prod_ts.index)
        wheat_prod_ts = wheat_prod_ts.rename(columns={"Value": "Total_Wheat_Prod"})
        st.dataframe(wheat_prod_ts)

        ### EXPLORATORY ANALYSIS
        st.markdown("### Exploratory Analysis")
        st.markdown("***Resampling & Aggregation***")
        st.markdown(
            "In order to have an visual glimpse of a time-series trend, it is common to resample a dataset & to aggregate as so:"
        )
        st.code(
            "df['Total_Wheat_Prod'].resample(rule='5A').agg(['mean','std','min','max'])"
        )
        st.markdown("Here you can choose how much you want to aggregate you df:")
        aggregation = st.selectbox(label="", options=["2Y", "5Y", "10Y", "15Y"])
        # Setting the .agg function to '5A' implies we aggregate every 5 years
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        aggregated = (
            wheat_prod_ts["Total_Wheat_Prod"]
            .resample(rule=aggregation)
            .agg(["mean", "std", "min", "max"])
        )
        aggregated["mean"].plot(label="Mean per Year")
        plt.fill_between(
            aggregated.index,
            aggregated["max"],
            aggregated["min"],
            alpha=0.5,
            label="min-max per year",
        )
        plt.legend()
        st.pyplot(fig)

        st.markdown(
            "*Resample & get **standard deviation every 2 years** to see how much wheat production is volatile*"
        )
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].resample("2A").std().plot()
        st.pyplot(fig)

        st.markdown(
            "_We can then plot a consolidated graph with an overview of the trend_"
        )
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].plot()
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].resample("2A").mean().plot(
            label="Mean every 2 years", lw=3, ls=":", alpha=0.8
        )
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].resample(
            "10A"
        ).mean().plot(label="Mean per Decade", lw=2, ls="--", alpha=0.8)
        plt.legend()
        plt.title("Evolution of the Total Wheat Production worldwide")
        st.pyplot(fig)

        ### STATISTICAL ANALYSIS
        st.markdown("### Statistical Analysis")
        st.markdown("**What do we seek to analyse?**")
        st.markdown(
            "Before applying any statistical model on a Time Series & forecast some predictions over it, our series has to be on a ***stationary state***, which means that, over different time periods,"
        )
        st.markdown(
            "- the mean should be a *constant* (visually parallel to the x-axis)"
        )
        st.markdown(
            "- the standard deviation should be a *constant* (visually parallel to the x-axis)"
        )
        st.markdown("- auto-covariance should not depend on time.")

        st.markdown("**Why does a time-series has to be stationary to be forecasted?**")
        st.markdown(
            "In case a time series has a particular behavior over a time interval, then there's a high probability that over a different interval, this behavior will be the same if the series is **stationary**. Thus it is very helpful to accurately forecast."
        )
        st.markdown("**What elements infer on this stationarity?**")
        st.markdown("Several elements play a role:")
        st.markdown(
            "- Trend: general direction of a time series over a certain period of time"
        )
        st.markdown(
            "- Seasonality: seasonal variances (often explained by external factors"
        )
        st.markdown(
            "- Noise: irregularity in the data represented by spikes & downs at random"
        )
        st.markdown("- Cyclicity: cyclic behavior of data")
        st.write("")
        st.write("")
        st.write("")

        ## Moving Average
        st.markdown("**_Moving Average_**")
        st.markdown(
            "In order to first analyse the mean, we could plot the overall average by taking each value & divide by *N*.But instead of averaging all the values at once, we will compute the mean of the values on a certain ***window*** that we will roll through our sample. By modifying the **window** parameter, we smooth more or less our mean since we add more or less values as the numerator, over the denominator which is the chosen number of periods to evaluate. This allows to erase **transitory fluctuations** & keep the **long-term trend**, but it also gives us a **visual hint** that our time series **is** or **is not** ***stationary***. "
        )
        st.markdown(
            "<img src='https://raw.githubusercontent.com/nicoboou/ml_eml/main/img/moving-averages.gif' width='700' align='center'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""The successive calculation of moving averages for the same sequence of numbers requires that all the values used by the previous averages be retained ${\bar {x}}_{n}$, in order to replace the oldest term $x_{n-\mathrm {N}}$ with the most recent term $x_{n}$."""
        )
        st.markdown(
            r"""${\displaystyle {\bar {x}}_{n}={\frac {1}{\mathrm {N} }}\ \displaystyle {\sum _{k=0}^{N-1}\;{x_{n-k}}}{\text{, or also }}{\bar {x}}_{n}={\bar {x}}_{n-1}+{\frac {x_{n}-x_{n-\mathrm {N} }}{\mathrm {N} }}}$"""
        )
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].plot()
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].rolling(
            window=2, center=True
        ).mean().plot(label="moving average centered", lw=3, ls="--", alpha=0.8)
        plt.legend()
        st.pyplot(fig)

        ## **Exponential Weighted Moving Average (EWMA)**
        st.markdown("**Exponential Weighted Moving Average (EWMA)**")
        st.markdown(
            "Compared to simple moving averages, EMAs give greater weight to recent (more relevant) data. Indeed in a standard moving average, the oldest price in a fixed series is dropped. By contrast, all prices in a chart influence an exponential moving average: **older prices gradually diminish in significance, this time *exponentially* (in contrast with Simple WMA)***."
        )
        st.markdown(
            r"""The EMA for a series ${\displaystyle T}$ may be calculated recursively:"""
        )
        st.markdown(
            r"""$${\displaystyle S_{t}={\begin{cases}T_{0},&t=0\\\alpha T_{t}+(1-\alpha )\cdot S_{t-1},&t>0\end{cases}}}$$"""
        )
        st.markdown("Where:")
        st.markdown(
            r"- The coefficient ${\displaystyle \alpha }$ represents the degree of weighting decrease, a constant smoothing factor between 0 and 1. A higher ${\displaystyle \alpha }$ discounts older observations faster."
        )
        st.markdown(
            r"- ${\displaystyle T_{t}}$ is the value at a time period ${\displaystyle t}$."
        )
        st.markdown(
            r"- ${\displaystyle S_{t}}$ is the value of the EMA at any time period ${\displaystyle t}$."
        )
        st.markdown(
            r"- ${\displaystyle S_{t}}$ is the value of the EMA at any time period ${\displaystyle t}$."
        )
        st.markdown("More concise way to write it:")
        st.markdown(
            r"$$\bar {x}_{t} = \sum_{n=0}^{+\infty} \alpha ({1-{\alpha}})^n {\bar {x}_{t-1}}$$"
        )

        fig, axs = plt.subplots(1, figsize=(15, 9), dpi=200)

        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].plot()
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].rolling(
            window="730D"
        ).mean().plot(label="simple moving average", lw=2, ls=":", alpha=0.8)
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].rolling(
            window="730D", center=True
        ).mean().plot(label="centered moving average", lw=2, ls=":", alpha=0.8)

        # Here is our new EWMA function
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].ewm(alpha=0.7).mean().plot(
            label="exponential weighted moving average", lw=2, ls="--"
        )
        plt.legend()
        st.pyplot(fig)

        st.markdown(
            r"Let's compare how the function behaves for different values of $\alpha$"
        )

        fig, axs = plt.subplots(1, figsize=(15, 9), dpi=200)
        wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].plot()
        for i in np.arange(0.1, 1, 0.2):
            wheat_prod_ts.loc["1961":"2019", "Total_Wheat_Prod"].ewm(
                alpha=i
            ).mean().plot(label=f"ewma {i}", lw=2, ls=":")
        plt.legend()
        st.pyplot(fig)

        st.markdown(
            r"***FIRST CONCLUSION:*** It seems that our Exponential Weighted Moving Average works best when ***$\alpha = 0.9$***"
        )
        st.markdown(
            "***SECOND CONCLUSION: it appears that our time series IS NOT stationary since our rolling mean & rolling std aren't contstant over time***. We will confirm this hypothesis by using the *Augmented Dickey-Fuller statistical test*."
        )

        st.write("")
        st.write("")
        ## Augmented Dickey-Fuller test
        st.markdown("**Augmented Dickey-Fuller Test**")
        st.markdown(
            'The *Augmented Dickey–Fuller test* is used in time serie analysis to gives us several values which will help in identifying *stationarity*. We first define the Null hypothesis has that *"a time-series is non-stationary*. Then a  Statistics test & some critical values for some confidence levels are defined & performed. If the test is less than the certain critical values, we can reject the null hypothesis & say that the series is stationary. The ADCF test also gives us a *p-value*. Acc to the null hypothesis, lower values of p is better.'
        )

        st.markdown("_Results of Augmented Dickey Fuller test:_")
        st.code("from statsmodels.tsa.stattools import adfuller")
        st.code("adfuller(wheat_prod_ts['Total_Wheat_Prod'], autolag='AIC'")
        dick_full_test = adfuller(wheat_prod_ts["Total_Wheat_Prod"], autolag="AIC")

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
        st.markdown("To be stationary: we should have got:")
        st.markdown("- the *p-value* as low as possible (p-value<0.5)")
        st.markdown(
            "- the *critical values* at the different confidence intervals should be as close as possible to the Test Statistic"
        )
        st.markdown(
            "**CONCLUSION:** we can see that our time-series is **NOT stationary**. Thus we need to perform some *data transformation* to it."
        )

        ### DATA TRANSFORMATION
        st.markdown("### Data Transformation")
        st.markdown(
            "To be able to perform our statistical analysis, we need to make our time-series stationary"
        )
        st.markdown("**LogScale our data**")
        st.markdown(
            "We first ScaleTransform our data points using the log function to flatten our spikes."
        )

        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        df_logscaled = np.log(wheat_prod_ts)
        plt.plot(df_logscaled)
        st.pyplot(fig)

        st.markdown("**Time Shift Transform**")
        st.code("df_logscaled_diff = df_logscaled - df_logscaled.shift()")
        df_logscaled_diff = df_logscaled - df_logscaled.shift()
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        plt.plot(df_logscaled_diff)
        st.pyplot(fig)

        df_logscaled_diff.dropna(inplace=True)
        stationarity_test(df_logscaled_diff, "Total_Wheat_Prod")

        st.markdown("Our model appears to be now **stationary**!")

        ### ARIMA
        st.markdown("### ARIMA (Univariate Statistical Analysis)")
        st.markdown(
            "To analyze a time-series & forecast its future datapoints, two methods are possible:"
        )
        st.markdown(
            "- **Univariate**: use of only pre-existing datapoints, and only this"
        )
        st.markdown(
            "- **Multivariate**: use of external variables correlated to the time-series, thus using multiple features of a same dataset"
        )
        st.markdown(
            "**ARIMA (_Auto Regressive Integrated Moving Average_)** is a **_univariate_** method for time-series analysis. It is a combination of 2 models AR(Auto Regressive) & MA(Moving Average). It has 3 hyperparameters - P(auto regressive lags),d(order of differentiation),Q(moving avg.) which respectively comes from the AR, I & MA components. The AR part is correlation between prev & current time periods. To smooth out the noise, the MA part is used. The I part binds together the AR & MA parts."
        )
        st.markdown("**How to find the best values P & Q ?**")
        st.markdown(
            "We need to take help of ACF(Auto Correlation Function) & PACF(Partial Auto Correlation Function) plots. ACF & PACF graphs are used to find value of P & Q for ARIMA. We need to check, for which value in x-axis, graph line drops to 0 in y-axis for 1st time."
        )
        st.markdown("- From PACF(at y=0), get P")
        st.markdown("- From ACF(at y=0), get Q")
        st.write("")
        st.write("")

        ## Arima Model
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        plt.plot(df_logscaled_diff)
        plt.plot(st.session_state.arima_model.fittedvalues, color="red")
        plt.title("Predictions of ARIMA model on Logscaled Shifted Dataset")
        st.pyplot(fig)

        # Predictions vs History
        predictions_ARIMA = from_pred_to_val(
            st.session_state.arima_model.fittedvalues, df_logscaled["Total_Wheat_Prod"]
        )

        # Forecast
        forecast_ARIMA = st.session_state.arima_model.fittedvalues.append(
            st.session_state.arima_model.forecast(steps=5)
        )
        forecast_ARIMA = from_pred_to_val(
            forecast_ARIMA, df_logscaled["Total_Wheat_Prod"]
        )
        st.markdown(
            "Let's have a look at some predicted values, after we ***descaled & deshifted*** them using ```log()```:"
        )
        st.dataframe(forecast_ARIMA.to_frame("Predictions (ARIMA) (tonnes)"))

        # Plot
        st.markdown("Let's now plot the **predictions** & the **forecast projections**")
        fig, axs = plt.subplots(2, figsize=(12, 12))
        # predictions vs history
        axs[0].plot(wheat_prod_ts)
        axs[0].plot(predictions_ARIMA, color="red")
        axs[0].set_title("Predictions -vs- History")

        # forecast
        axs[1].plot(wheat_prod_ts.loc["2005":"2020"], label="History")
        axs[1].plot(
            forecast_ARIMA.loc["2005":"2020"], color="orange", label="Predictions"
        )
        axs[1].plot(
            forecast_ARIMA.loc["2020":], color="green", ls="--", lw=2, label="Forecast"
        )
        axs[1].set_title("Forecast")
        plt.legend()
        st.pyplot(fig)

        ### RANDOM FOREST (UNIVARIATE)
        st.markdown("### Random Forests (Univariate Machine Learning method)")
        st.markdown(
            "Still using ***univariate*** method, Machine Learning has proven to be really powerful in time-series analyis. What I wanted to experiment is if such _new ML methods_ are **more powerful** than _traditional statistical_ ones ?"
        )
        st.markdown("**Transform our _time-series_ to a _supervised_ dataset**")
        st.markdown(
            "Such problem need that we transform our time-series into a _'supervised_'dataset. For that purpose, we need to build a new dataframe & input as features the ${n-1}$ years of the ```target_variable```we want to predict."
        )
        st.markdown("")

        st.code(code_block_rf, language="python")
        (
            expanded_wheat_prod,
            X,
            y,
            X_train,
            X_valid,
            y_train,
            y_valid,
        ) = from_series_to_supervised(wheat_prod_ts)
        st.write(X_train.shape)
        st.markdown(
            "Here, we are using the ***5*** preceding years to build our new supervised df."
        )
        st.markdown("**Model Training & Results**")
        st.markdown(
            "After splitting our new dataset into ```train```and ```test```sets, I performed some ***cross validation*** using **GridSearchCV** and picked the best RandomForestRegressor among many. Here are the results:"
        )
        print_score(st.session_state.rf_uni, X_train, y_train, X_valid, y_valid)

        st.markdown("**Predictions**")
        y_pred = st.session_state.rf_uni.predict(X)
        st.dataframe(pd.Series(y_pred).to_frame("Predictions (RF Univariate)"))

        # Plot
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        plt.plot(wheat_prod_ts, ls=":", label="History")
        plt.plot(wheat_prod_ts.index[5:], y_pred, "r", ls="--", label="Predictions")
        plt.title("Predictions with Random Forests")
        plt.legend()
        st.pyplot(fig)

        st.markdown("**Forecast**")
        rf_uni_forecasted = rf_forecast(
            st.session_state.rf_uni, expanded_wheat_prod, 10
        )
        st.dataframe(rf_uni_forecasted.tail(15))

        # Plot
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        plt.plot(wheat_prod_ts, ls=":", label="History")
        plt.plot(wheat_prod_ts.index[5:], y_pred, "r", ls="--", label="Predictions")
        plt.plot(
            rf_uni_forecasted["Target_Year"]["2019":], "s", ls=":", label="Forecasts"
        )
        plt.title("Predictions with Random Forests")
        plt.legend()
        st.pyplot(fig)
        st.markdown(
            "**CONCLUSION: It appears that our RandomForest Regressor has some troubles to extrapolate and forecast beyond the training data time range.**"
        )

        ### LINEAR REGRESSION (UNIVARIATE)
        st.markdown(
            "In order to ***extrapolate*** more, one algorithm could be very well fitted for the job: **linear regression**\ Let's have a look at the metrics when we train a simple LinearRegression algorithm:"
        )
        print_score(st.session_state.lr_uni, X_train, y_train, X_valid, y_valid)
        y_pred = st.session_state.lr_uni.predict(X)
        linear_forecast = rf_forecast(st.session_state.lr_uni, expanded_wheat_prod, 10)
        # Plot
        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        plt.plot(wheat_prod_ts, ls=":", label="History")
        plt.plot(wheat_prod_ts.index[5:], y_pred, "r", ls="--", label="Predictions")
        plt.plot(
            linear_forecast["Target_Year"]["2019":], "v", ls=":", label="Forecasts"
        )
        plt.title("Predictions with Linear Regression")
        plt.legend()
        st.pyplot(fig)

    elif prediction == "USA Wheat Yield (Multivariate)":
        st.markdown(
            "## II. Time-series Analysis #2: USA Wheat Yield Forecast using Earth Temp & Pop data"
        )
        st.markdown(
            "In the first experiment, I used only the ***univariate*** methodology, relying on past data only. Let's now see how the **multivariate** methods come into play for time-series forecasting."
        )
        st.markdown(
            "Using _multivariate_ methodology implies that other features, ***external*** to our _target_ variable, are used. I chose to rely on **Climate Metrics** & **Population statistics** to predict the Wheat Yield in the US."
        )
        st.markdown(
            "**Let's consolidate the data to get only the total USA population per year since 1950**"
        )
        st.markdown(
            "**_Note_**:In order to manipulate dates in a simpler manner, let's first tweak our 'Time' column in order to be able to put it in a DateTimeIndex format"
        )
        st.code(code_block_pop_df, language="python")
        st.dataframe(st.session_state.usa_pop)

        st.markdown(
            "We can now consolidate our 3 dataframes into 1: ```pop.csv```, ```crops_all.csv``` & ```climate_usa.csv```"
        )

        # Wheat Yield dataset
        wheat_yield_ts = st.session_state.crops_all[
            (st.session_state.crops_all["Item"] == "Wheat")
            & (st.session_state.crops_all["Element"] == "Yield")
        ].drop(
            columns=[
                "Area Code",
                "Item Code",
                "Item",
                "Element Code",
                "Unit",
            ]
        )

        # let's deflate our df to put all our "Year" columns in a single column
        wheat_yield_ts = wheat_yield_ts.melt(
            id_vars=["Continent", "Area", "Element"],
            var_name="Year",
            value_name="Value",
        )

        # Some small slicing to get the Year
        wheat_yield_ts["Year"] = wheat_yield_ts["Year"].apply(lambda x: x[1:])

        # Fill NaN values with 0 to be able to plot
        wheat_yield_ts["Value"] = wheat_yield_ts["Value"].fillna(0)
        wheat_yield_ts = pd.pivot_table(
            wheat_yield_ts, values="Value", index=["Year"], aggfunc=np.sum
        )
        wheat_yield_ts.index = pd.to_datetime(wheat_yield_ts.index)
        wheat_yield_ts = wheat_yield_ts.rename(columns={"Value": "Total_Wheat_YIELD"})
        usa_yield = wheat_yield_ts.merge(
            st.session_state.usa_pop,
            how="right",
            left_index=True,
            right_index=True,
            suffixes=(False, False),
        )
        usa_yield = usa_yield.merge(
            st.session_state.climate_usa, how="right", left_index=True, right_index=True
        )
        st.dataframe(usa_yield.tail())

        # Dataset Prep
        st.markdown("**Dataset Transform**")
        usa_yield = usa_yield.drop(columns=["LocID", "Location", "VarID", "MidPeriod"])
        usa_yield = usa_yield.melt(
            id_vars=[
                "Total_Wheat_YIELD",
                "PopMale",
                "PopFemale",
                "PopTotal",
                "PopDensity",
                "Variant",
                "Precipitations (historic)",
                "Precipitations (RCP2.6)",
                "Precipitations (RCP6)",
                "Precipitations (RCP8.5)",
            ],
            var_name="Temperature Type",
            value_name="Temperature (°C)",
            ignore_index=False,
        )
        usa_yield = usa_yield.melt(
            id_vars=[
                "Total_Wheat_YIELD",
                "PopMale",
                "PopFemale",
                "PopTotal",
                "PopDensity",
                "Variant",
                "Temperature Type",
                "Temperature (°C)",
            ],
            var_name="Precipitations Type",
            value_name="Precipitations (mm/y)",
            ignore_index=False,
        )

        # Historical data
        st.markdown("*Historical data*")
        st.code(
            """usa_yield_hist=usa_yield[(usa_yield["Temperature Type"] == "Temperature (historic)")& (usa_yield["Precipitations Type"] == "Precipitations (historic)")].loc["1961":"2019"]"""
        )
        usa_yield_hist = usa_yield[
            (usa_yield["Temperature Type"] == "Temperature (historic)")
            & (usa_yield["Precipitations Type"] == "Precipitations (historic)")
        ].loc["1961":"2019"]

        # Data we will use to forecast USA Wheat Yield
        st.markdown("*Data we will use to forecast USA Wheat Yield*")
        st.code(
            """usa_yield_preds=usa_yield[(usa_yield["Temperature Type"] != "Temperature (historic)") & (usa_yield["Precipitations Type"] != "Precipitations (historic)")].loc["2020":]"""
        )
        usa_yield_preds = usa_yield[
            (usa_yield["Temperature Type"] != "Temperature (historic)")
            & (usa_yield["Precipitations Type"] != "Precipitations (historic)")
        ].loc["2020":]

        ### RANDOM FOREST (MULTIVARIATE)

        st.markdown("### Random Forest (Multivariate")
        # Create our train_test_plit & variables
        predictors = usa_yield_hist.drop(
            columns=[
                "Total_Wheat_YIELD",
                "Variant",
                "Temperature Type",
                "Precipitations Type",
            ]
        )
        target = usa_yield_hist["Total_Wheat_YIELD"]
        X_train, X_valid, y_train, y_valid = train_test_split(
            predictors, target, test_size=0.3, random_state=42
        )
        st.markdown("*Shape of our train & test sets*")
        st.write(X_train.shape, X_valid.shape)
        st.write(y_train.shape, y_valid.shape)

        st.markdown(
            "**Metrics of our RandomForestRegressor using *multivariate* method:**"
        )
        print_score(st.session_state.rf_multi, X_train, y_train, X_valid, y_valid)
        st.info("*This model is doing **very well** !*")
        st.markdown(
            "**And hat are the most _'important'_ features in our model ? What weighted most in its decision ?**"
        )
        features_importance = features_importances(X_train, st.session_state.rf_multi)
        features_importance = pd.DataFrame.from_dict(
            features_importance, orient="index"
        )
        st.dataframe(features_importance)
        st.info(
            "We clearly see that the metrics related to *Population* are the **most influential ones** on our model"
        )

        full_pred = st.session_state.rf_multi.predict(predictors)

        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        plt.plot(usa_yield_hist["Total_Wheat_YIELD"], label="Historical")
        plt.plot(usa_yield_hist.index, full_pred, label="Predicted")
        plt.title("USA Wheat Yield")
        plt.legend()
        st.pyplot(fig)
        st.info(
            "**Prediction** on *historical data* appears very **well fitted** when plotted"
        )

        st.markdown("**Forecasting future datapoints**")
        predictors_to_forecast = usa_yield_preds.drop(
            columns=[
                "Total_Wheat_YIELD",
                "Variant",
                "Temperature Type",
                "Precipitations Type",
            ]
        )
        target_to_forecast = usa_yield_preds["Total_Wheat_YIELD"]
        forecast_multi_rf = st.session_state.rf_multi.predict(predictors_to_forecast)
        usa_yield_preds["Total_Wheat_YIELD"] = forecast_multi_rf
        st.dataframe(usa_yield_preds["Total_Wheat_YIELD"])
        options_temp = st.selectbox(
            "Temperature scenarios",
            usa_yield_preds["Temperature Type"].unique(),
        )
        options_precipitations = st.selectbox(
            "Precipitations scenarios",
            usa_yield_preds["Precipitations Type"].unique(),
        )
        options_pop = st.selectbox(
            "Population scenarios",
            usa_yield_preds["Variant"].unique(),
        )

        fig, axs = plt.subplots(1, figsize=(15, 7), dpi=120)
        plt.plot(usa_yield_hist["Total_Wheat_YIELD"], label="Historical")
        plt.plot(usa_yield_hist.index, full_pred, label="Predicted")
        plot_forecast_scenarios(
            usa_yield_preds, options_temp, options_precipitations, options_pop
        )
        plt.title("USA Wheat Yield Forecast (Random Forest Multivariate)")
        plt.legend()
        st.pyplot(fig)
        st.markdown("**CONCLUSIONS:**")
        st.markdown(
            "-  Even using multivariate method for time-series analysis, _RandomForests_ appears to **struggle with extropolation/forecasting**"
        )
        st.markdown(
            "- Nevertheless though, some specific scenarios seem to influence the model enough to make the trend curb. These scenarios mostly depend on **Population** forecasting."
        )

        ### LINEAR REGRESSION (MULTIVARIATE)
        st.markdown("### Linear Regression (Multivariate")
        st.markdown(
            "**What are the result metrics for a Linear Regression model trained using multiple **external** features ?**"
        )
        forecast_multi_lr = st.session_state.lr_multi.predict(predictors_to_forecast)
        usa_yield_preds["Total_Wheat_YIELD"] = forecast_multi_lr
        print_score(st.session_state.lr_multi, X_train, y_train, X_valid, y_valid)
        st.markdown(
            "Our $RMSE$ and $R^2$ appear to be as performant as the ones we got from our multivariate RF regressor"
        )
        options_temp2 = st.selectbox(
            "Temperature scenarios",
            np.sort(usa_yield_preds["Temperature Type"].unique())[::-1],
        )
        options_precipitations2 = st.selectbox(
            "Precipitations scenarios",
            np.sort(usa_yield_preds["Precipitations Type"].unique())[::-1],
        )
        options_pop2 = st.selectbox(
            "Population scenarios",
            np.sort(usa_yield_preds["Variant"].unique())[::-1],
        )
        # Plot
        fig, axs = plt.subplots(1, figsize=(17, 8), dpi=200)
        plt.plot(usa_yield_hist["Total_Wheat_YIELD"], label="Historical")
        plt.plot(usa_yield_hist.index, full_pred, label="Predicted")
        plot_forecast_scenarios(
            usa_yield_preds,
            "Temperature (RCP2.6)",
            "Precipitations (RCP2.6)",
            "Zero migration",
        )
        plot_forecast_scenarios(
            usa_yield_preds, "Temperature (RCP8.5)", "Precipitations (RCP8.5)", "Low"
        )
        plot_forecast_scenarios(
            usa_yield_preds, "Temperature (RCP2.6)", "Precipitations (RCP8.5)", "High"
        )
        plot_forecast_scenarios(
            usa_yield_preds,
            options_temp2,
            options_precipitations2,
            options_pop2,
        )
        plt.title("USA Wheat Yield Forecast (Linear Regression Multivariate)")
        plt.legend()
        st.pyplot(fig)

        ### CONCLUSION
        st.markdown("### Conclusion")
        st.markdown(
            "Of course these predictions are to be put into perspective and context. Climate change, and more recently Ukrainian war dismantle such algorithms. Such hazards, natural and man-made disasters lead to huge changes in food production. And it is undermining our ability to predict crop yields. Even the best models break down in this context."
        )
        st.markdown(
            "1. **Univariate analysis** is ***less powerful*** than **multivariate analysis** for our dataset"
        )
        st.markdown(
            "2. **Machine Learning** techniques (RandomForest, LR) appear ***more precise*** & ***less time-sensitive*** than **traditional statistical methods** (e.g. ARIMA here)**"
        )
        st.markdown(
            "3. In terms of ***prediction*** (predicting historical data), RandomForests are better."
        )
        st.markdown(
            "4. But in terms of ***forecasting*** (projecting the future), linear regression is **more adapted** since RandomForest isn't capable to ***extropolate*** well (i.e. predict outside original time range from training data)"
        )
