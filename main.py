# main.py

import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import sim1  # Importing the simulation module

# Define the available runs
available_runs = [56000, 40000, 25000]

# Run Monte Carlo simulations with specified runs
results_by_immigration = sim1.monte_carlo_simulations(10, net_migration_values=available_runs)

# Dash app initialization
app = dash.Dash(__name__)

# Updated Layout with Dropdown
app.layout = html.Div([
    # Dropdown for selecting runs
    html.Div([
        html.Label("Select Run"),
        dcc.Dropdown(
            id='run-dropdown',
            options=[{'label': str(run), 'value': run} for run in available_runs],
            value=56000,  # Default selected run
            clearable=False
        )
    ], style={'width': '200px', 'padding': '10px'}),

    # Top row with three charts
    html.Div([
        html.Div([
            dcc.Graph(
                id="population-breakdown",
                config={"displayModeBar": False},
                style={"height": "300px"}
            )
        ], style={"width": "32%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(
                id="immigrant-percentage",
                config={"displayModeBar": False},
                style={"height": "300px"}
            )
        ], style={"width": "32%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(
                id="fertility-chart",
                config={"displayModeBar": False},
                style={"height": "300px"}
            )
        ], style={"width": "32%", "display": "inline-block"})
    ], style={"display": "flex", "justifyContent": "space-between"}),

    # Middle row with two larger charts side by side
    html.Div([
        html.Div([
            dcc.Graph(
                id="population-trend",
                config={"displayModeBar": False}
            )
        ], style={"width": "48%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(
                id="age-sex-pyramid",
                config={"displayModeBar": False}
            )
        ], style={"width": "48%", "display": "inline-block"})
    ], style={"display": "flex", "justifyContent": "space-between"}),

    # Bottom row with the histograms
    html.Div([
        dcc.Graph(
            id="immigrant-gene-histogram",
            config={"displayModeBar": False}
        )
    ], style={"width": "48%", "display": "inline-block"})
])

# Callback for population trend graph
@app.callback(
    Output("population-trend", "figure"),
    [Input("run-dropdown", "value"),
     Input("population-trend", "hoverData")]
)
def create_population_trend(selected_run, _):
    # Extract data based on the selected run
    stats = results_by_immigration[selected_run]["avg_stats"]
    min_stats = results_by_immigration[selected_run]["min_stats"]
    max_stats = results_by_immigration[selected_run]["max_stats"]
    simulation_batch = results_by_immigration[selected_run]["avg_simulation_batch"]
    
    years = list(range(len(stats)))
    total_population = [stat['total_population'] * simulation_batch for stat in stats]
    min_total_population = [stat['total_population'] * simulation_batch for stat in min_stats]
    max_total_population = [stat['total_population'] * simulation_batch for stat in max_stats]
    
    fig = go.Figure()

    # Average line
    fig.add_trace(go.Scatter(
        x=years,
        y=total_population,
        mode="lines+markers",
        name="Total Population (Avg)",
        line=dict(color="green", width=3)
    ))

    # Min line (invisible)
    fig.add_trace(go.Scatter(
        x=years,
        y=min_total_population,
        mode="lines",
        line=dict(color="green", width=0),
        name="Min Population",
        showlegend=False
    ))

    # Max line with fill area to min
    fig.add_trace(go.Scatter(
        x=years,
        y=max_total_population,
        mode="lines",
        fill='tonexty',  # fill area between this trace and previous trace
        line=dict(color="green", width=0),
        name="Max Population",
        showlegend=False
    ))

    fig.update_layout(
        title="Total Population Over Time (with Min/Max)",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Total Population"),
        hovermode="x unified"
    )

    return fig

# Callback for population breakdown graph
@app.callback(
    Output("population-breakdown", "figure"),
    [Input("run-dropdown", "value"),
     Input("population-breakdown", "hoverData")]
)
def create_population_breakdown(selected_run, _):
    # Extract data based on the selected run
    stats = results_by_immigration[selected_run]["avg_stats"]
    simulation_batch = results_by_immigration[selected_run]["avg_simulation_batch"]
    
    years = list(range(len(stats)))
    native_population = [stat['native_population'] * simulation_batch for stat in stats]
    immigrant_1_population = [stat['immigrant_1_population'] * simulation_batch for stat in stats]
    immigrant_2_population = [stat['immigrant_2_population'] * simulation_batch for stat in stats]
    mixed_population = [stat['mixed_population'] * simulation_batch for stat in stats]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=native_population,
        mode="lines",
        name="Native Population",
        line=dict(dash='dash', color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=years,
        y=immigrant_1_population,
        mode="lines",
        name="Immigrant 1 Population",
        line=dict(dash='dot', color='red')
    ))
    fig.add_trace(go.Scatter(
        x=years,
        y=immigrant_2_population,
        mode="lines",
        name="Immigrant 2 Population",
        line=dict(dash='dashdot', color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=years,
        y=mixed_population,
        mode="lines",
        name="Mixed Population",
        line=dict(color='purple')
    ))

    fig.update_layout(
        title="Population Breakdown Over Time",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Population"),
        hovermode="x unified"
    )

    return fig

# Callback for immigrant percentage graph
@app.callback(
    Output("immigrant-percentage", "figure"),
    [Input("run-dropdown", "value"),
     Input("immigrant-percentage", "hoverData")]
)
def create_immigrant_percentage(selected_run, _):
    # Extract data based on the selected run
    stats = results_by_immigration[selected_run]["avg_stats"]
    
    years = list(range(len(stats)))
    immigrant_percentage = [stat['immigrant_percentage'] for stat in stats]  # percentages can be averaged directly
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=immigrant_percentage,
        mode="lines",
        name="Immigrant & Mixed Percentage",
        line=dict(color='orange')
    ))

    fig.update_layout(
        title="Immigrant & Mixed Percentage Over Time",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Percentage (%)"),
        hovermode="x unified"
    )

    return fig

# Callback for age-sex pyramid
@app.callback(
    Output("age-sex-pyramid", "figure"),
    [Input("run-dropdown", "value"),
     Input("population-trend", "hoverData")]
)
def update_age_sex_pyramid(selected_run, hoverData):
    # Determine the hovered year
    if hoverData and "points" in hoverData:
        year = hoverData["points"][0]["x"]
    else:
        year = 0  # Default to the first year if no hover data is available

    pyramid_data = results_by_immigration[selected_run]["avg_population_data"][year]["pyramid_data"]
    simulation_batch = results_by_immigration[selected_run]["avg_simulation_batch"]

    # Scale counts by simulation_batch (no negation)
    scaled_native_male_counts = [count * simulation_batch for count in pyramid_data[1]]
    scaled_immigrant_male_counts = [count * simulation_batch for count in pyramid_data[2]]
    scaled_native_female_counts = [count * simulation_batch for count in pyramid_data[3]]
    scaled_immigrant_female_counts = [count * simulation_batch for count in pyramid_data[4]]

    # Calculate max count for axis scaling
    max_pyramid_count = 300000  # You might want to adjust this dynamically

    fig = go.Figure()

    # Plot natives first to have immigrants on the outside
    # Males
    fig.add_trace(go.Bar(
        y=pyramid_data[0],
        x=scaled_native_male_counts,
        orientation="h",
        name="Native Males",
        marker_color="blue",
        hovertemplate='Age Group: %{y}<br>Count: %{x}<extra></extra>',
        offsetgroup=-1,
        legendgroup='Males'
    ))
    fig.add_trace(go.Bar(
        y=pyramid_data[0],
        x=scaled_immigrant_male_counts,
        orientation="h",
        name="Immigrant Males",
        marker_color="lightblue",
        hovertemplate='Age Group: %{y}<br>Count: %{x}<extra></extra>',
        offsetgroup=-1,
        base=scaled_native_male_counts,
        legendgroup='Males'
    ))
    # Females
    fig.add_trace(go.Bar(
        y=pyramid_data[0],
        x=scaled_native_female_counts,
        orientation="h",
        name="Native Females",
        marker_color="red",
        hovertemplate='Age Group: %{y}<br>Count: %{x}<extra></extra>',
        offsetgroup=1,
        legendgroup='Females'
    ))
    fig.add_trace(go.Bar(
        y=pyramid_data[0],
        x=scaled_immigrant_female_counts,
        orientation="h",
        name="Immigrant Females",
        marker_color="lightsalmon",
        hovertemplate='Age Group: %{y}<br>Count: %{x}<extra></extra>',
        offsetgroup=1,
        base=scaled_native_female_counts,
        legendgroup='Females'
    ))

    fig.update_layout(
        title=f"Age-Sex Pyramid for Year {year}",
        xaxis=dict(
            title="Population",
            tickvals=[-max_pyramid_count, -max_pyramid_count/2, 0, max_pyramid_count/2, max_pyramid_count],
            ticktext=[str(int(abs(val))) for val in [-max_pyramid_count, -max_pyramid_count/2, 0, max_pyramid_count/2, max_pyramid_count]],
            range=[-max_pyramid_count * 1.1, max_pyramid_count * 1.1],  # Adding some padding
        ),
        yaxis=dict(title="Age Group", autorange="reversed"),
        barmode="relative",
        bargap=0.1,
        hovermode="y unified",
        legend_title="Population Groups"
    )

    return fig

# Callback for immigrant gene histogram
@app.callback(
    Output("immigrant-gene-histogram", "figure"),
    [Input("run-dropdown", "value"),
     Input("population-trend", "hoverData")]
)
def update_immigrant_gene_histogram(selected_run, hoverData):
    # Determine the hovered year
    if hoverData and "points" in hoverData:
        year = hoverData["points"][0]["x"]
    else:
        year = 0  # Default to the first year if no hover data is available

    gene_values = results_by_immigration[selected_run]["avg_population_data"][year]["gene_values"]  # List of dicts with gene percentages
    simulation_batch = results_by_immigration[selected_run]["avg_simulation_batch"]

    # Extract gene percentages for each gene type
    native_values = [gv.get('native', 0) for gv in gene_values]
    immigrant1_values = [gv.get('immigrant_1', 0) for gv in gene_values]
    immigrant2_values = [gv.get('immigrant_2', 0) for gv in gene_values]

    # Compute histograms manually
    bins = np.arange(0, 110, 10)  # 0-10, 10-20, ..., 90-100
    hist_native, _ = np.histogram(native_values, bins=bins)
    hist_immigrant1, _ = np.histogram(immigrant1_values, bins=bins)
    hist_immigrant2, _ = np.histogram(immigrant2_values, bins=bins)

    # Scale counts by simulation_batch to reflect the entire population
    hist_native_scaled = hist_native * simulation_batch
    hist_immigrant1_scaled = hist_immigrant1 * simulation_batch
    hist_immigrant2_scaled = hist_immigrant2 * simulation_batch

    # Create histogram traces for each gene type
    fig = go.Figure()

    # Plot Native gene histogram
    fig.add_trace(go.Bar(
        x=bins[:-1] + 5,  # Bin centers: 5, 15, ..., 95
        y=hist_native_scaled,
        name="Native",
        marker_color="blue",
        opacity=0.8
    ))

    # Plot Immigrant 1 gene histogram
    fig.add_trace(go.Bar(
        x=bins[:-1] + 5,
        y=hist_immigrant1_scaled,
        name="Immigrant 1",
        marker_color="red",
        opacity=0.8
    ))

    # Plot Immigrant 2 gene histogram
    fig.add_trace(go.Bar(
        x=bins[:-1] + 5,
        y=hist_immigrant2_scaled,
        name="Immigrant 2",
        marker_color="orange",
        opacity=0.8
    ))

    # Update layout to display histograms side by side
    fig.update_layout(
        title=f"Immigrant Gene Value Distribution for Year {year}",
        xaxis=dict(
            title="Gene Percentage (%)",
            tickvals=bins[:-1] + 5,
            ticktext=[f"{int(val)}-{int(val+10)}%" for val in bins[:-1]],
            range=[0, 100],
        ),
        yaxis=dict(
            title="Number of Individuals",
            range=[0, max(hist_native_scaled.max(), hist_immigrant1_scaled.max(), hist_immigrant2_scaled.max()) * 1.1]
        ),
        barmode='group',  # Changed from 'overlay' to 'group' for side-by-side bars
        hovermode="x unified",
        legend=dict(title="Gene Types")
    )

    return fig

# Callback for the fertility chart
@app.callback(
    Output("fertility-chart", "figure"),
    [Input("run-dropdown", "value"),
     Input("fertility-chart", "hoverData")]
)
def update_fertility_chart(selected_run, _):
    # Extract data based on the selected run
    stats = results_by_immigration[selected_run]["avg_stats"]
    
    years = list(range(len(stats)))
    avg_children_per_female_natives = [stat['avg_children_per_female_natives'] for stat in stats]
    avg_children_per_female_immigrants = [stat['avg_children_per_female_immigrants'] for stat in stats]
    avg_children_per_female_mixed = [stat['avg_children_per_female_mixed'] for stat in stats]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=avg_children_per_female_natives,
        mode="lines",
        name="Native Females",
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=years,
        y=avg_children_per_female_immigrants,
        mode="lines",
        name="Immigrant Females",
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=years,
        y=avg_children_per_female_mixed,
        mode="lines",
        name="Mixed Females",
        line=dict(color='purple')
    ))
    fig.update_layout(
        title="Average Children per Female Over Lifetime",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Average"),
        hovermode="x unified"
    )
    return fig

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
