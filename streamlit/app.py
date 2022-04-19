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

# for app
import streamlit as st

# for plots
import plotly.express as px

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

    ### LOAD AGRICULTURAL & CROPS DATA ###

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

    # 2. Load Gas Emissions dataset
    url = "https://raw.githubusercontent.com/nicoboou/ml_eml/main/data/emissions/emissions_full.csv"
    emissions_df = pd.read_csv(url, sep=",", encoding="latin-1", low_memory=False)
    co2_df = emissions_df.drop(emissions_df[emissions_df["Code Élément"] != 7273].index)

    # 3. Load Population dataset
    pop_df_raw = pd.read_csv(
        "https://raw.githubusercontent.com/nicoboou/ml_eml/main/data/pop_data_onu.csv",
        thousands=" ",
    )

    # load regression model
    # path_to_model = os.path.join(path_to_repo, "saves", "RF_regressor.pk")
    # with open(path_to_model, "rb") as file:
    # model = dill.load(file)

    # load images
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
    st.session_state.emissions_df = emissions_df
    st.session_state.co2_df = co2_df
    st.session_state.imgs = imgs
    # st.session_state.n_valid = n_valid
    # st.session_state.model = model


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
        options=["Agricultural & Crops", "Gas Emissions", "Population"],
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
        st.write("")

    elif dataset == "Gas Emissions":
        st.header("Gas Emissions Dataset")
        st.subheader("Summary")
        st.write(
            "The FAOSTAT domain Emissions Totals disseminates information estimates of CH4, N2O and CO2 emissions/removals and their aggregates in CO2eq in units of kilotonnes (kt, or 106 kg)."
        )
        st.subheader("Description")
        st.markdown(
            "The FAOSTAT domain Emissions Totals summarizes the greenhouse gas (GHG) emissions disseminated in the FAOSTAT Climate Change Emissions domains, generated from agriculture and forest land. They consist of methane (CH4), nitrous oxide (N2O) and carbon dioxide (CO2) emissions from crop and livestock activities, forest management and include land use and land use change processes. Data are computed at Tier 1 of the IPCC Guidelines for National greenhouse gas (GHG) Inventories (IPCC, 1996; 1997; 2000; 2002; 2006; 2014). Estimates are available by country, with global coverage for the period 1961–2019 with projections for 2030 and 2050 for some categories of emissions or 1990–2019 for others. The database is updated annually."
        )
        st.subheader("Main Sample")
        st.dataframe(st.session_state.emissions_df)
        st.subheader("Link")
        st.write(
            "Source: Organisation des Nations Unies pour l'alimentation et l'agriculture (FAO)"
        )
        st.write("")

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
        st.subheader("Link")
        st.write("Source: ")
        st.write("")


# EXPLORATION
elif page == "Exploration":
    explore = st.sidebar.selectbox(
        label="Select dataset to explore",
        options=["Agricultural & Crops", "Gas Emissions", "Population"],
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

    elif explore == "Gas Emissions":
        tmp_df = co2_df

    elif explore == "Population":
        tmp_df = pop_df_raw


# PREDICTIONS
