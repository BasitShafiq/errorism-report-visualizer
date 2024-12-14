import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import folium
from plotly import graph_objs as go
from fpdf import FPDF
import tempfile
from folium.plugins import FastMarkerCluster, Fullscreen
from datetime import datetime
from selenium import webdriver
import plotly.io as pio
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Streamlit page configuration
st.set_page_config(page_title="GTD EXPLORER")

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

# Sidebar header
st.sidebar.header('GTDE')
st.sidebar.subheader('GLOBAL TERRORISM DATABASE EXPLORER')

# Load and clean data
data = pd.read_csv("Global Terrorism Cleaned with missing dates.csv", encoding='latin1')
data.replace(r'\s+', np.nan, regex=True)
data['DATE'] = pd.to_datetime(data['DATE']).dt.date
data['DEAD'] = data['DEAD'].astype('Int64')
data['INJURED'] = data['INJURED'].astype('Int64')

# Overall stats calculation
overall_attacks = len(data)
overall_regions = data['REGION'].nunique()
overall_countries = data['COUNTRY'].nunique()
overall_perpetrators = data['PERPETRATOR'].nunique()
overall_deaths = data['DEAD'].sum()
overall_injuries = data['INJURED'].sum()

# Display overall stats
st.markdown("<h2 style='text-align: center;'>Overall Stats</h2>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-evenly; padding: 20px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px;">
        <div style="text-align: center;"><strong>Attacks:</strong><br>{overall_attacks}</div>
        <div style="text-align: center;"><strong>Regions:</strong><br>{overall_regions}</div>
        <div style="text-align: center;"><strong>Countries:</strong><br>{overall_countries}</div>
        <div style="text-align: center;"><strong>Perpetrators:</strong><br>{overall_perpetrators}</div>
        <div style="text-align: center;"><strong>Deaths:</strong><br>{int(overall_deaths)}</div>
        <div style="text-align: center;"><strong>Injuries:</strong><br>{int(overall_injuries)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar filters
col1, col2 = st.sidebar.columns(2)
fromdate = col1.date_input(
    "FROM", value=(pd.Timestamp("1970-02-07").date()), 
    min_value=(pd.Timestamp(min(data['DATE'])).date()), 
    max_value=(pd.Timestamp(max(data['DATE'])).date())
)
todate = col2.date_input(
    "TO", value=(pd.Timestamp(max(data['DATE'])).date()), 
    min_value=(pd.Timestamp(min(data['DATE'])).date()), 
    max_value=(pd.Timestamp(max(data['DATE'])).date()))

# Region, country, city, perpetrator, category filters
regions = list(data["REGION"].sort_values().unique())
regions = [x for x in regions if pd.isnull(x) == False]
regions.insert(0, "All")
region = st.sidebar.multiselect("REGION", regions, default="All")

countries = list(data["COUNTRY"].sort_values().unique())
countries = [x for x in countries if pd.isnull(x) == False]
countries.insert(0, "All")
country = st.sidebar.multiselect("COUNTRY", countries, default="All")

cities = list(data["CITY"].sort_values().unique())
cities = [x for x in cities if pd.isnull(x) == False]
cities.insert(0, "All")
city = st.sidebar.multiselect("CITY", cities, default="All")

perpetrators = list(data["PERPETRATOR"].sort_values().unique())
perpetrators = [x for x in perpetrators if pd.isnull(x) == False]
perpetrators.insert(0, "All")
perpetrator = st.sidebar.multiselect("PERPETRATOR", perpetrators, default="All")

categories = list(data["CATEGORY"].sort_values().unique())
categories = [x for x in categories if pd.isnull(x) == False]
categories.insert(0, "All")
category = st.sidebar.multiselect("CATEGORY", categories, default="All")

# Apply filters
mask = (data['DATE'] > np.datetime64(fromdate)) & (
    data['DATE'] <= np.datetime64(todate))
data = data.loc[mask]

if "All" not in region:
    mask = data["REGION"].isin(region)
    data = data[mask]
if "All" not in country:
    mask = data["COUNTRY"].isin(country)
    data = data[mask]
if "All" not in city:
    mask = data["CITY"].isin(city)
    data = data[mask]
if "All" not in perpetrator:
    mask = data["PERPETRATOR"].isin(perpetrator)
    data = data[mask]
if "All" not in category:
    mask = data["CATEGORY"].isin(category)
    data = data[mask]


st.sidebar.title("Current Statistics")
st.sidebar.text("Attacks: "+str(len(data)) +
                "\nRegions: "+str(len(data["REGION"].unique()))+ "\nCountries: "+str(len(data["COUNTRY"].unique()))+"\nPerpetrators: "+str(len(data["PERPETRATOR"].unique()))+"\nDeaths: "+str(int(data["DEAD"].sum()))+"\nInjuries: "+str(int(data["INJURED"].sum())))


# Filter coordinates for valid values
st.markdown("<h2 style='text-align: center;'>Map Overview</h2>", unsafe_allow_html=True)
filtered_data = data[data['COORDINATES'].notnull()]  # Remove rows with null COORDINATES
lat = []
lon = []
for coord in filtered_data['COORDINATES']:
    try:
        lat_val, lon_val = map(float, coord.split(","))
        if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:  # Check if within valid ranges
            lat.append(lat_val)
            lon.append(lon_val)
    except:
        continue

callback = """\
function (row) {
    var marker;
    marker = L.circle(new L.LatLng(row[0], row[1]), {color:'red'});
    return marker;
};
"""
make_map_responsive = """ 
<style>
[title~="st.iframe"] { width: 100%}
</style>
"""

st.markdown(make_map_responsive, unsafe_allow_html=True)

folium_map = folium.Map(tiles='cartodbpositron')

FastMarkerCluster(data=list(zip(lat, lon)), callback=callback).add_to(folium_map)
Fullscreen().add_to(folium_map)

folium_static(folium_map, width=1000)

# Pie chart section
st.markdown("<h2 style='text-align: center;'>Pie Chart</h2>", unsafe_allow_html=True)
freq = st.radio("FREQUENCY", ('DEAD', 'INJURED'), horizontal=True)
dist = st.radio("DISTRIBUTION", ('CATEGORY', 'PERPETRATOR', 'REGION', 'COUNTRY', 'STATE', 'CITY'), horizontal=True)

# Create the pie chart
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9c27b0', '#03a9f4']
fig1 = go.Figure(data=[go.Pie(
    labels=list(data[freq].groupby(data[dist]).sum().sort_values().nlargest(12).index),
    values=list(data[freq].groupby(data[dist]).sum().sort_values().nlargest(12)),
    marker=dict(colors=colors)
)])
fig1.update_traces(
    hoverinfo='label+percent', 
    textinfo='value', 
    textfont_size=20,
    marker=dict(line=dict(color='#000000', width=1))
)
fig1.update_layout(title="FULL", showlegend=True)

# Display first pie chart with fullscreen
st.plotly_chart(fig1, use_container_width=True, config={
    'displayModeBar': True,
    'modeBarButtonsToAdd': ['toggleFullscreen']
})

# Graphs for DEAD and INJURED
st.subheader("DEAD")
fig_dead = go.Figure()
fig_dead.add_trace(go.Scatter(x=data['DATE'], y=data['DEAD'].groupby(data['DATE']).sum()))
st.plotly_chart(
    fig_dead,
    use_container_width=True,
    config={
        'displayModeBar': True, 
        'modeBarButtonsToAdd': ['toggleSpikelines', 'zoomIn2d', 'zoomOut2d', 'resetScale2d'],
        'toImageButtonOptions': {'format': 'svg', 'filename': 'Dead_Graph'}, 
        'displaylogo': False 
    }
)

st.subheader("INJURED")
fig_injured = go.Figure()
fig_injured.add_trace(go.Scatter(x=data['DATE'], y=data['INJURED'].groupby(data['DATE']).sum()))
st.plotly_chart(
    fig_injured,
    use_container_width=True,
    config={
        'displayModeBar': True, 
        'modeBarButtonsToAdd': ['toggleSpikelines', 'zoomIn2d', 'zoomOut2d', 'resetScale2d'],
        'toImageButtonOptions': {'format': 'svg', 'filename': 'Injured_Graph'}, 
        'displaylogo': False 
    }
)

# Display filtered data as table and enable CSV download
st.markdown("<h2 style='text-align: center;'>Database Table</h2>", unsafe_allow_html=True)
st.dataframe(data)

# Convert DataFrame to CSV
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Create CSV download button
csv = convert_df(data)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv'
)


####################################################################################################

####################################################################################################

####################################################################################################

####################################################################################################

####################################################################################################




# Function to capture screenshots
def capture_screenshot(url, element_id, output_path):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    element = driver.find_element("id", element_id)
    element.screenshot(output_path)
    driver.quit()

# Function to generate a PDF report
def generate_pdf(title, overview_stats, current_stats, map_image_path, pie_chart_path, graph_paths, table_data, output_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")

    # Overall Stats
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, "Overall Stats:", ln=True)
    for stat, value in overview_stats.items():
        pdf.cell(0, 10, f"{stat}: {value}", ln=True)

    # Current Stats
    pdf.ln(10)
    pdf.cell(0, 10, "Current Stats:", ln=True)
    for stat, value in current_stats.items():
        pdf.cell(0, 10, f"{stat}: {value}", ln=True)

    # Map
    pdf.ln(10)
    pdf.cell(0, 10, "Map Overview:", ln=True)
    pdf.image(map_image_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(80)

    # Pie Chart
    pdf.cell(0, 10, "Pie Chart:", ln=True)
    pdf.image(pie_chart_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(80)

    # Graphs
    for graph_path in graph_paths:
        pdf.cell(0, 10, f"Graph:", ln=True)
        pdf.image(graph_path, x=10, y=pdf.get_y(), w=180)
        pdf.ln(80)

    # Table
    pdf.cell(0, 10, "Filtered Data:", ln=True)
    pdf.set_font("Courier", "", 10)
    for index, row in table_data.iterrows():
        row_string = ', '.join([str(x) for x in row])
        pdf.cell(0, 10, row_string, ln=True)

    pdf.output(output_path)

# Function to generate an HTML report
def generate_html_report(title, overview_stats, current_stats, map_image_path, pie_chart_path, graph_paths, table_data, output_path):
    html_content = f"""
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            h1, h2 {{ text-align: center; }}
            .section {{ margin-bottom: 30px; }}
            .stats {{ display: flex; justify-content: space-evenly; }}
            .map, .pie-chart, .graph {{ text-align: center; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p style="text-align:center;">This report provides an overview of terrorism data, including overall and current statistics, graphical insights, and a detailed table.</p>

        <div class="section">
            <h2>Overall Stats</h2>
            <div class="stats">
                {''.join([f'<div><strong>{k}</strong><br>{v}</div>' for k, v in overview_stats.items()])}
            </div>
        </div>

        <div class="section">
            <h2>Current Stats</h2>
            <div class="stats">
                {''.join([f'<div><strong>{k}</strong><br>{v}</div>' for k, v in current_stats.items()])}
            </div>
        </div>

        <div class="section map">
            <h2>Map Overview</h2>
            <img src="{map_image_path}" alt="Map Overview" style="width:100%; max-w'<div id="map_id">',th:600px;">
        </div>

        <div class="section pie-chart">
            <h2>Pie Chart</h2>
            <img src="{pie_chart_path}" alt="Pie Chart" style="width:100%; max-width:600px;">
        </div>

        <div class="section">
            <h2>Graphs</h2>
            {''.join([f'<div class="graph"><img src="{path}" alt="Graph" style="width:100%; max-width:600px;"></div>' for path in graph_paths])}
        </div>

        <div class="section">
            <h2>Filtered Data Table</h2>
            <table border="1" style="width:100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        {''.join([f'<th>{col}</th>' for col in table_data.columns])}
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'<tr>{"".join([f"<td>{val}</td>" for val in row])}</tr>' for _, row in table_data.iterrows()])}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    with open(output_path, 'w') as f:
        f.write(html_content)

# Updated Streamlit App for additional functionality
st.title("Enhanced Report Generator")

# Add a "Download Report" button
if st.button("Download Report (PDF & CSV)"):
    with tempfile.TemporaryDirectory() as tmpdirname:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Capture screenshots
        map_image_path = f"{tmpdirname}/map.png"
        capture_screenshot("http://localhost:8501", "map_id", map_image_path)

        pie_chart_path = f"{tmpdirname}/pie_chart.png"
        capture_screenshot("http://localhost:8501", "pie_chart_id", pie_chart_path)

        graph_paths = []
        for i, graph_id in enumerate(["graph1_id", "graph2_id"]):
            graph_path = f"{tmpdirname}/graph_{i}.png"
            capture_screenshot("http://localhost:8501", graph_id, graph_path)
            graph_paths.append(graph_path)

        # Generate PDF
        pdf_path = f"{tmpdirname}/report_{timestamp}.pdf"
        generate_pdf(
            title="Global Terrorism Report",
            overview_stats={
                "Attacks": overall_attacks,
                "Regions": overall_regions,
                "Countries": overall_countries,
                "Perpetrators": overall_perpetrators,
                "Deaths": overall_deaths,
                "Injuries": overall_injuries,
            },
            current_stats={
                "Filtered Attacks": len(data),
                "Regions": len(data["REGION"].unique()),
                "Countries": len(data["COUNTRY"].unique()),
                "Deaths": int(data["DEAD"].sum()),
                "Injuries": int(data["INJURED"].sum()),
            },
            map_image_path=map_image_path,
            pie_chart_path=pie_chart_path,
            graph_paths=graph_paths,
            table_data=data,
            output_path=pdf_path,
        )

        # Generate HTML
        html_path = f"{tmpdirname}/report_{timestamp}.html"
        generate_html_report(
            title="Global Terrorism Report",
            overview_stats={
                "Attacks": overall_attacks,
                "Regions": overall_regions,
                "Countries": overall_countries,
                "Perpetrators": overall_perpetrators,
                "Deaths": overall_deaths,
                "Injuries": overall_injuries,
            },
            current_stats={
                "Filtered Attacks": len(data),
                "Regions": len(data["REGION"].unique()),
                "Countries": len(data["COUNTRY"].unique()),
                "Deaths": int(data["DEAD"].sum()),
                "Injuries": int(data["INJURED"].sum()),
            },
            map_image_path=map_image_path,
            pie_chart_path=pie_chart_path,
            graph_paths=graph_paths,
            table_data=data,
            output_path=html_path,
        )

        # Display download links
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name=f"report_{timestamp}.pdf",
                mime="application/pdf"
            )

        with open(html_path, "r") as html_file:
            st.download_button(
                label="Download HTML Report",
                data=html_file,
                file_name=f"report_{timestamp}.html",
                mime="text/html"
            )

# Add a "Download CSV" button for full dataset
csv_full = convert_df(data)
st.download_button(
    label="Download Full CSV",
    data=csv_full,
    file_name="full_dataset.csv",
    mime="text/csv"
)
