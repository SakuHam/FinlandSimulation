# main.py

import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import sim1  # Importing the simulation module

# Run the simulation once
(
    stats,
    max_count,
    max_hist_count,
    avg_children_per_female_natives,
    avg_children_per_female_immigrants,
    avg_children_per_female_mixed,
    population_data,
    simulation_batch  # Include simulation_batch in the outputs
) = sim1.run_large_simulation()

# Scale max_count and max_hist_count
max_count *= simulation_batch
max_hist_count *= simulation_batch

# Prepare data for plotting
years = list(range(len(stats)))
total_population = [stat['total_population'] * simulation_batch for stat in stats]
native_population = [stat['native_population'] * simulation_batch for stat in stats]
immigrant_population = [stat['immigrant_population'] * simulation_batch for stat in stats]
mixed_population = [stat['mixed_population'] * simulation_batch for stat in stats]
immigrant_percentage = [stat['immigrant_percentage'] for stat in stats]  # Percentages remain the same

# Dash app initialization
app = dash.Dash(__name__)

# Updated Layout
app.layout = html.Div([
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
    ], style={"display": "flex", "justify-content": "space-between"}),

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
    ], style={"display": "flex", "justify-content": "space-between"}),

    # Bottom row with the histogram
    html.Div([
        dcc.Graph(
            id="immigrant-gene-histogram",
            config={"displayModeBar": False}
        )
    ], style={"width": "48%", "display": "inline-block"})
])

# Callback for total population trend graph
@app.callback(
    Output("population-trend", "figure"),
    Input("population-trend", "hoverData")
)
def create_population_trend(_):
    # Total population trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=total_population,
        mode="lines+markers",
        name="Total Population",
        line=dict(color="green", width=3)
    ))

    fig.update_layout(
        title="Total Population Over Time",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Total Population"),
        hovermode="x unified"
    )

    return fig

# Callback for population breakdown graph
@app.callback(
    Output("population-breakdown", "figure"),
    Input("population-breakdown", "hoverData")
)
def create_population_breakdown(_):
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
        y=immigrant_population,
        mode="lines",
        name="Immigrant Population",
        line=dict(dash='dot', color='red')
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
    Input("immigrant-percentage", "hoverData")
)
def create_immigrant_percentage(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=immigrant_percentage,
        mode="lines",
        name="Immigrant Percentage",
        line=dict(color='orange')
    ))

    fig.update_layout(
        title="Immigrant & Mixed Percentage Over Time",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Percentage (%)"),
        hovermode="x unified"
    )

    return fig

# Updated Callback for age-sex pyramid
@app.callback(
    Output("age-sex-pyramid", "figure"),
    Input("population-trend", "hoverData")
)
def update_age_sex_pyramid(hoverData):
    # Determine the hovered year
    if hoverData and "points" in hoverData:
        year = hoverData["points"][0]["x"]
    else:
        year = 0  # Default to the first year if no hover data is available

    pyramid_data = population_data[year]["pyramid_data"]

    # Scale counts by simulation_batch (no negation)
    scaled_native_male_counts = [count * simulation_batch for count in pyramid_data["native_male_counts"]]
    scaled_immigrant_male_counts = [count * simulation_batch for count in pyramid_data["immigrant_male_counts"]]
    scaled_native_female_counts = [count * simulation_batch for count in pyramid_data["native_female_counts"]]
    scaled_immigrant_female_counts = [count * simulation_batch for count in pyramid_data["immigrant_female_counts"]]

    # Calculate max count for axis scaling
    max_male = max([abs(count) for count in scaled_native_male_counts + scaled_immigrant_male_counts])
    max_female = max([abs(count) for count in scaled_native_female_counts + scaled_immigrant_female_counts])
    max_pyramid_count = max(max_male, max_female)

    fig = go.Figure()

    # Plot natives first to have immigrants on the outside
    # Males (negative counts)
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=scaled_native_male_counts,
        orientation="h",
        name="Native Males",
        marker_color="blue",
        hovertemplate='Age Group: %{y}<br>Count: %{x}<extra></extra>',
        offsetgroup=-1,
        legendgroup='Males'
    ))
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=scaled_immigrant_male_counts,
        orientation="h",
        name="Immigrant Males",
        marker_color="lightblue",
        hovertemplate='Age Group: %{y}<br>Count: %{x}<extra></extra>',
        offsetgroup=-1,
        base=scaled_native_male_counts,
        legendgroup='Males'
    ))
    # Females (positive counts)
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=scaled_native_female_counts,
        orientation="h",
        name="Native Females",
        marker_color="red",
        hovertemplate='Age Group: %{y}<br>Count: %{x}<extra></extra>',
        offsetgroup=1,
        legendgroup='Females'
    ))
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
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
    Input("population-trend", "hoverData")
)
def update_immigrant_gene_histogram(hoverData):
    # Determine the hovered year
    if hoverData and "points" in hoverData:
        year = hoverData["points"][0]["x"]
    else:
        year = 0  # Default to the first year if no hover data is available

    gene_values = population_data[year]["gene_values"]

    # Create histogram
    bins = np.arange(0, 1.1, 0.1)  # Bins from 0 to 1 in steps of 0.1
    hist, bin_edges = np.histogram(gene_values, bins=bins)

    # Scale histogram counts
    hist = hist * simulation_batch

    # Prepare data for bar chart
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist,
        width=0.08,  # Adjust width to avoid bars touching
        marker_color='purple'
    ))

    fig.update_layout(
        title=f"Immigrant Gene Value Distribution for Year {year}",
        xaxis=dict(title="Immigrant Gene Value"),
        yaxis=dict(title="Number of Individuals", range=[0, max_hist_count]),
        hovermode="x"
    )

    return fig

# Callback for the fertility chart
@app.callback(
    Output("fertility-chart", "figure"),
    Input("population-trend", "hoverData")
)
def update_fertility_chart(hoverData):
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
