import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import folium
import folium.plugins as plugins
import json
import requests

#chicago_crime dataset from: https://data.cityofchicago.org/Public-Safety/Crimes-2023/xguy-4ndq
#chicago-community-areas.geojson from: https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6

chicago_crime = pd.read_csv('chicago_crime.csv')
comm_data = pd.read_csv('comm_areas.csv')

chicago_crime = pd.merge(chicago_crime, comm_data[['AREA_NUMBE', 'COMMUNITY']], left_on='Community Area', right_on='AREA_NUMBE')
chicago_crime.rename(columns={'COMMUNITY': 'Community Area Name'}, inplace=True)

def top_10_crimes():
    top_crimes = chicago_crime.groupby("Primary Type").size().reset_index(name='Total Count')
    top_crimes = top_crimes.sort_values('Total Count', ascending=False).head(10)

    # Create bar plot using Seaborn
    sns.set(style='whitegrid')
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x='Primary Type', y='Total Count', data=top_crimes, color='#4F4F4F')

    # Set plot title and axis labels
    ax.set_title('Top 10 Most Common Crimes in Chicago (2023)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Type of Crime', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Crimes', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate and align x-axis labels
    plt.tight_layout()

    plt.show()

def crime_rate_over_time():
    results = chicago_crime.groupby('Date').size().reset_index(name='Total Count')

    results['Date'] = pd.to_datetime(results['Date'])
    results.set_index('Date', inplace=True)

    daily_crime = results.resample('D').sum()

    monthly_crime = daily_crime.groupby(pd.Grouper(freq='M')).sum()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_crime.index, monthly_crime['Total Count'], color='#4F4F4F')

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))

    plt.title('Chicago Crime Rate over Time', fontsize=18, fontweight='bold')
    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Crimes', fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', labelsize=12)

    plt.show()

def top_10_dangerous_neighborhoods():
    results = chicago_crime.groupby(['Community Area', 'Community Area Name']).size().reset_index(name='Total Count')
    results = results[results['Community Area'] != '']
    results['Community Area'] = results['Community Area'].astype(int)
    results.sort_values(by='Total Count', ascending=False, inplace=True)
    results = results.head(10)

    results = results.reset_index().merge(comm_data[['COMMUNITY', 'the_geom']], left_on='Community Area Name', right_on='COMMUNITY')
    results['longitude'] = results['the_geom'].apply(lambda x: float(x.replace('MULTIPOLYGON (((', '').replace(')))', '').split()[1][:-1]))
    results['latitude'] = results['the_geom'].apply(lambda x: float(x.replace('MULTIPOLYGON (((', '').replace(')))', '').split()[1][:-1]))

    # Create a map object and add a heatmap layer with the property crime rate data
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10)

    with open('chicago-community-areas.geojson') as f:
        geojson_data = json.load(f)

    # Create Choropleth layer
    folium.Choropleth(
        geo_data=geojson_data,
        name='Choropleth',
        data=results,
        columns=['Community Area Name', 'Total Count'],
        key_on='feature.properties.community',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Dangerous Communities in Chicago (# of crime)',).add_to(m)

    folium.plugins.HeatMap(results[['latitude', 'longitude', 'Total Count']].values, radius=15, max_zoom=13).add_to(m)

    m.save('chicago_dangerous_community_heatmap.html')

    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(results['Community Area Name'], results['Total Count'], color='#4F4F4F')
    ax.set_title('Top 10 Most Dangerous Communities in Chicago', fontsize=18, fontweight='bold')
    ax.set_xlabel('Community Area', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Crimes', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()

def domestic_violence_incidents():
    keywords = ['DOMESTIC', 'ASSAULT']
    results = chicago_crime[chicago_crime['Primary Type'].isin(['BATTERY', 'ASSAULT', 'CRIMINAL SEXUAL ASSAULT', 'OFFENSE INVOLVING CHILDREN'])]
    results = results[results['Description'].str.contains('|'.join(keywords), case=False)]
    results['Date'] = pd.to_datetime(results['Date'], format='%m/%d/%Y %I:%M:%S %p')
    results = results[(results['Date'] >= '2023-01-01') & (results['Date'] <= '2023-04-25')]
    results = results.sort_values('Date')

    monthly_counts = results.groupby(results['Date'].dt.strftime('%B'))['Date'].count()
    monthly_counts = monthly_counts.reindex(['January', 'February', 'March', 'April'])
    count = monthly_counts.sum()

    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))
    ax = monthly_counts.plot(kind='bar', color='#4F4F4F')
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Assaults', fontsize=14, fontweight='bold')
    ax.set_title('Domestic Assault by Month (Total: {})'.format(count), fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, fontsize=8)
    plt.subplots_adjust(bottom=0.2)

    plt.show()

def gun_violence_crimes():

    results = chicago_crime[(chicago_crime['Primary Type'].isin(['ASSAULT', 'BATTERY'])) &
                            (chicago_crime['Description'].str.contains('GUN', case=False)) &
                            (chicago_crime['Description'].str.contains('AGGRAVATED', case=False))] 
    results['Date'] = pd.to_datetime(results['Date'], format='%m/%d/%Y %I:%M:%S %p')
    results = results[(results['Date'] >= '2023-01-01') & (results['Date'] <= '2023-04-25')]
    results = results.sort_values('Date')                 

    monthly_counts = results.groupby(results['Date'].dt.strftime('%B'))['Date'].count()
    monthly_counts = monthly_counts.reindex(['January', 'February', 'March', 'April'])
    count = monthly_counts.sum()

    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))
    ax = monthly_counts.plot(kind='bar', color='#4F4F4F')
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Gun Violence Crime', fontsize=14, fontweight='bold')
    ax.set_title('Gun Violence Crimes by Month (Total: {})'.format(count),fontsize=18, fontweight='bold')

    plt.xticks(rotation=45, fontsize=8)
    plt.subplots_adjust(bottom=0.2)

    plt.show()

def neighborhoods_highest_property_crime():
    query = '''
        SELECT 
            "Community Area",
            ROUND(CAST(SUM(CASE WHEN "Primary Type" = 'THEFT' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) AS "Property Crime Rate"
        FROM 
            chicago_crime
        WHERE 
            "Primary Type" = 'THEFT'
        GROUP BY 
            "Community Area"
        ORDER BY 
            "Property Crime Rate" DESC
        LIMIT 10
    '''

    # Execute SQL query on pandas DataFrame
    prop_crime_rate = chicago_crime.query('`Primary Type` == "THEFT"') \
        .groupby('Community Area Name')['ID'].agg(['count']) \
        .join(chicago_crime.groupby('Community Area Name')['ID'].agg(['count']), rsuffix='_total') \
        .assign(prop_rate=lambda x: round(x['count'] / x['count_total'] * 100, 2)) \
        .sort_values('prop_rate', ascending=False) \
        .head(10)

    prop_crime_rate = prop_crime_rate.reset_index().merge(comm_data[['COMMUNITY', 'the_geom']], left_on='Community Area Name', right_on='COMMUNITY')
    prop_crime_rate['longitude'] = prop_crime_rate['the_geom'].apply(lambda x: float(x.replace('MULTIPOLYGON (((', '').replace(')))', '').split()[1][:-1]))
    prop_crime_rate['latitude'] = prop_crime_rate['the_geom'].apply(lambda x: float(x.replace('MULTIPOLYGON (((', '').replace(')))', '').split()[1][:-1]))

    # Create a map object and add a heatmap layer with the property crime rate data
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10)

    with open('chicago-community-areas.geojson') as f:
        geojson_data = json.load(f)

    # Create Choropleth layer
    folium.Choropleth(
        geo_data=geojson_data,
        name='Choropleth',
        data=prop_crime_rate,
        columns=['Community Area Name', 'prop_rate'],
        key_on='feature.properties.community',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Property Crime Rate (%)',).add_to(m)

    folium.plugins.HeatMap(prop_crime_rate[['latitude', 'longitude', 'prop_rate']].values, radius=15, max_zoom=13).add_to(m)

    m.save('chicago_property_crime_heatmap.html')

    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(prop_crime_rate['Community Area Name'], prop_crime_rate['prop_rate'], color='#4F4F4F')
    ax.set_title('Chicago Community Areas with the Highest Property Crime Rates', fontsize=18, fontweight='bold')
    ax.set_xlabel('Community Area Name', fontsize=14, fontweight='bold')
    ax.set_ylabel('Property Crime Rate (%)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.tick_params(axis='x', labelrotation=45)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()

def top_10_crimes_of_2022():

    url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json?$where=date BETWEEN '2022-01-01T00:00:00' and '2022-04-25T23:59:59'"

    response = requests.get(url)

    json_data = response.json()

    df = pd.DataFrame(json_data)

    top_crimes = df.groupby("primary_type").size().reset_index(name='total_count')

    # Sort the data in descending order by total_count and get the top 10
    top_crimes = top_crimes.sort_values('total_count', ascending=False).head(10)

    # Create bar plot using Seaborn
    sns.set(style='whitegrid')
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x='primary_type', y='total_count', data=top_crimes, color='#4F4F4F')

    # Set plot title and axis labels
    ax.set_title('Top 10 Most Common Crimes in Chicago (2022)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Type of Crime', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Crimes', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate and align x-axis labels
    plt.tight_layout()
    plt.show()

# Main program
def prompt_user():
    while True:
        print("Please select a query to run:")
        print("1. Top 10 crimes committed in Chicago (2023)")
        print("2. Crime rate over time in Chicago")
        print("3. Top 10 dangerous neighborhoods in Chicago")
        print("4. Domestic violence incidents in Chicago")
        print("5. Gun violence crimes in Chicago")
        print("6. Neighborhoods with highest property crime in Chicago")
        print("7. Top 10 crimes committed in Chicago (2022)")
        print("Type 'quit' to exit the program")

        query_num = input("Enter the number of the query you want to run: ")

        if query_num == '1':
            top_10_crimes()
        elif query_num == '2':
            crime_rate_over_time()
        elif query_num == '3':
            top_10_dangerous_neighborhoods()
        elif query_num == '4':
            domestic_violence_incidents()
        elif query_num == '5':
            gun_violence_crimes()
        elif query_num == '6':
            neighborhoods_highest_property_crime()
        elif query_num == '7':
            top_10_crimes_of_2022()
        elif query_num.lower() == 'quit':
            break
        else:
            print("Invalid input. Please enter a number between 1 and 6, or type 'quit' to exit.")

        # Check if any figures are open
        if plt.get_fignums():
            try:
                plt.close('all')
            except matplotlib.cbook.MatplotlibDeprecationWarning:
                pass

if __name__ == "__main__":
    prompt_user()