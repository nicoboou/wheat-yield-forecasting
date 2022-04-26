# Time-Series Forecasting

Time-series Forecasting using different methods:

- Univariate method:
  - ARIMA Model (Statistical analysis)
  - Random Forest (Machine Learning)
  - Linear Regression (Machine Learning)
- Multivariate method:
  - Random Forest (Machine Learning)
  - Linear Regression (Machine Learning)

## TLDR

- Streamlit app: install `requirements.txt` and then `streamlit run app.py`
- Jupyter Notebook: https://github.com/nicoboou/time-series-forecasting/blob/main/notebook/Crops%20Yield%20Notebook.ipynb

## Project Outline

As part of a student project, we were given a white card to :

- find a reliable and complete dataset
- clean this dataset and do the necessary data preprocessing to be able to do machine learning
- do data mining to find potentially interesting insights for the future
- use our knowledge of machine learning to perform an accurate analysis
- highlighting the results obtained with Streamlit

## Dataset

### Description

Crop statistics for 173 products in Africa, America, Asia, Europe and Oceania, collected from 1961 to 2019.

### Summary

Data from the Food and Agriculture Organization of the United Nations (FAO)

Achieving food security for all is at the heart of FAO's efforts - ensuring that people have regular access to enough quality food to lead active and healthy lives.
Our three main objectives are: the eradication of hunger, food insecurity and malnutrition; the eradication of poverty and the promotion of economic and social progress for all; and the sustainable management and use of natural resources, including land, water, air, climate and genetic resources, for the benefit of present and future generations.

Primary crops, fibre crops. Crop statistics are recorded for 173 commodities, covering the following categories: Primary crops, Primary fibre crops, Cereals, Secondary cereals, Citrus, Fruit, Jute and related fibres, Oilcake equivalent, Primary oilseeds, Dry vegetables, Roots and tubers, Green fruits and vegetables and Melons. Data are expressed in terms of area harvested, quantity produced, yield and quantity of seed. The aim is to provide comprehensive coverage of production of all primary crops for all countries and regions of the world.

_Source_ : Organisation des Nations Unies pour l'alimentation et l'agriculture (FAO)

## Notebook Structure

You will find the following structure in the .ipynb file:

1. Data preprocessing
2. Exploratory Data Analysis
3. Time-series Analysis & Forecasting
   - Univariate method
   - Multivariate method

## Requirements

The file `requirements.txt`will allow you to create your dedicated `conda env`without any missing dependency.
Just type `conda create --name <env_name> --file requirements.txt` in your terminal
