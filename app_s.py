import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

# Title and Introduction
st.title("Global CO2 Emissions Analysis")
st.markdown("""
This Streamlit app visualizes global fossil fuel emissions data by country and year, focusing on key greenhouse gases (CO2, methane, nitrous oxide, total GHG).
""")

# Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("visualizing_global_co2_data.csv")
    except:
        path = "https://drive.google.com/file/d/16w_qjmvXFkPcR7tt4W1UKQC7ZIwgK8sR/view?usp=sharing" 
        path = 'https://drive.google.com/uc?id=' + path.split('/')[-2]
        df = pd.read_csv(path)
    return df

df = load_data()

# Clean and prepare data
columns_to_keep = [
    'iso_code', 'year', 'population', 'gdp', 'co2', 'methane', 'nitrous_oxide',
    'total_ghg'
]
df = df[columns_to_keep]
df_countries = df[df['iso_code'].notna()]

# Sidebar for country selection
iso_to_country = df.drop_duplicates(subset=['iso_code']).set_index('iso_code')['country'].to_dict()
iso_to_country_clean = {k: v for k, v in iso_to_country.items() if pd.notna(k)}
country_options = list(iso_to_country_clean.keys())
# COUNTRY = st.sidebar.selectbox("Select a country for distribution plots", country_options, index=country_options.index('USA') if 'USA' in country_options else 0)
# country_name = iso_to_country_clean[COUNTRY]
COUNTRY = "USA"  # Default country for the example

# 1. Distribution plots for each feature (for selected country)
st.header(f"Feature Distributions for {COUNTRY}")
import matplotlib.pyplot as plt

for column in ['population', 'gdp', 'co2', 'methane', 'nitrous_oxide', 'total_ghg']:
    plt.figure(figsize=(8, 3))
    data = df[(df['iso_code'] == COUNTRY) & df[column].notna() & (df[column] != 0)][column]
    sns.histplot(data, bins=30, kde=True, color='blue')
    plt.xlabel(column)
    plt.ylabel('Number of Records')
    plt.title(f'Distribution of {column} in {country_name}')
    st.pyplot(plt.gcf())
    plt.close()

# 2. Worldwide average trend over years
st.header("Worldwide Average Feature Trends Over Years")
for column in ['population', 'gdp', 'co2', 'methane', 'nitrous_oxide', 'total_ghg']:
    plt.figure(figsize=(8, 3))
    data = df_countries[df_countries[column].notna() & (df_countries[column] != 0)]
    sns.lineplot(
        data=data,
        x='year',
        y=column,
        estimator='mean',
        marker='o'
    )
    plt.xlabel('Year')
    plt.ylabel(column)
    plt.title(f'Worldwide Average {column} Trend Over Years')
    st.pyplot(plt.gcf())
    plt.close()

# 3. Correlation heatmap
st.header("Correlation Heatmap of Numerical Features")
corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
st.pyplot(plt.gcf())
plt.close()

# 4. Top 20 countries by cumulative CO2 emissions
st.header("Top 20 Countries by Cumulative CO2 Emissions")
cumulative_ghg = df.groupby('iso_code', as_index=False)['co2'].sum()
top20_ghg = cumulative_ghg.sort_values('co2', ascending=False).head(20)
top20_ghg['country'] = top20_ghg['iso_code'].map(iso_to_country_clean)
plt.figure(figsize=(10, 5))
sns.barplot(data=top20_ghg, x='country', y='co2', palette='viridis')
plt.xticks(rotation=75)
plt.xlabel('Country')
plt.ylabel('Cumulative CO2 Emissions')
plt.title('Top 20 Countries by CO2 Emissions')
st.pyplot(plt.gcf())
plt.close()

# 5. Top 20 countries by cumulative CO2 emissions per capita (population threshold)
st.header("Top 20 Countries by Cumulative CO2 Emissions Per Capita")
POP_THRESHOLD = 60_000
df['total_co2_per_capita'] = df.apply(
    lambda row: row['co2'] / row['population'] if pd.notnull(row['co2']) and pd.notnull(row['population']) and row['population'] != 0 else None,
    axis=1
)
cumulative_ghg_per_capita = df.groupby('iso_code', as_index=False)['total_co2_per_capita'].sum()
top20_ghg_per_capita = cumulative_ghg_per_capita.sort_values('total_co2_per_capita', ascending=False).head(20)
top20_ghg_per_capita['country'] = top20_ghg_per_capita['iso_code'].map(iso_to_country_clean)
avg_pop = df.groupby('iso_code')['population'].mean()
valid_iso_codes = avg_pop[avg_pop > POP_THRESHOLD].index
filtered_top20 = top20_ghg_per_capita[top20_ghg_per_capita['iso_code'].isin(valid_iso_codes)]
filtered_top20 = filtered_top20.sort_values('total_co2_per_capita', ascending=False).head(20)
plt.figure(figsize=(10, 5))
sns.barplot(
    data=filtered_top20,
    x='country',
    y='total_co2_per_capita',
    palette='viridis'
)
plt.xticks(rotation=75)
plt.xlabel('Country')
plt.ylabel('Cumulative CO2 Emissions Per Capita')
plt.title(f'Top 20 Countries by Cumulative CO2 Emissions Per Capita (Population > {POP_THRESHOLD})')
st.pyplot(plt.gcf())
plt.close()