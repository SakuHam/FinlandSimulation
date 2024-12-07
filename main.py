# main.py

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import sim1  # Your simulation module with init and run_one_year functions
import copy

# Initialize the simulation
sim1.init_large_simulation()

# Create a Dash app
app = dash.Dash(__name__)

# We'll store all simulation outputs in a dcc.Store so that the state
# persists between callbacks without using global variables.
app.layout = html.Div([
    html.Button("Advance Year", id="advance-year-button", n_clicks=0),
    dcc.Store(id='simulation-data', data={
        "stats": [],
        "max_count": 0,
        "max_hist_count": 0,
        "avg_children_per_female_natives": [],
        "avg_children_per_female_immigrants": [],
        "avg_children_per_female_mixed": [],
        "population_data": {},
        "simulation_batch": sim1.simulation_batch
    }),

    # The layout from your original code, without the initial simulation run
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

    # Bottom row with the histograms
    html.Div([
        dcc.Graph(
            id="immigrant-gene-histogram",
            config={"displayModeBar": False}
        )
    ], style={"width": "48%", "display": "inline-block"})
])


# Callback to run one year of simulation and update the data store
@app.callback(
    Output("simulation-data", "data"),
    Input("advance-year-button", "n_clicks"),
    State("simulation-data", "data")
)
def advance_year(n_clicks, data):
    if n_clicks > 0:
        # Run one year of simulation
        (stat, year_max_count, year_max_hist_count,
         avg_nat, avg_imm, avg_mix,
         year_population_data, simulation_batch) = sim1.run_large_simulation()

        # Update data in store
        data["stats"].append(stat)
        data["avg_children_per_female_natives"].append(avg_nat)
        data["avg_children_per_female_immigrants"].append(avg_imm)
        data["avg_children_per_female_mixed"].append(avg_mix)

        # Update max_count and max_hist_count if needed
        if year_max_count > data["max_count"]:
            data["max_count"] = year_max_count
        if year_max_hist_count > data["max_hist_count"]:
            data["max_hist_count"] = year_max_hist_count

        # The index of the current year is len(stats)-1
        current_year = len(data["stats"]) - 1
        data["population_data"][current_year] = year_population_data

    return data

def generate_derived_data(data):
    """Helper function to generate arrays needed for graphs from the store data."""
    stats = data["stats"]
    simulation_batch = data["simulation_batch"]

    years = list(range(len(stats)))
    if not stats:
        # Return 10 empty or default values
        return (years, [], [], [], [], [], [], [], [], [])

    total_population = [stat['total_population'] * simulation_batch for stat in stats]
    native_population = [stat['native_population'] * simulation_batch for stat in stats]
    immigrant_1_population = [stat['immigrant_1_population'] * simulation_batch for stat in stats]
    immigrant_2_population = [stat['immigrant_2_population'] * simulation_batch for stat in stats]
    mixed_population = [stat['mixed_population'] * simulation_batch for stat in stats]
    immigrant_percentage = [stat['immigrant_percentage'] for stat in stats]

    avg_children_per_female_natives = data["avg_children_per_female_natives"]
    avg_children_per_female_immigrants = data["avg_children_per_female_immigrants"]
    avg_children_per_female_mixed = data["avg_children_per_female_mixed"]

    return (years, total_population, native_population, immigrant_1_population,
            immigrant_2_population, mixed_population, immigrant_percentage,
            avg_children_per_female_natives, avg_children_per_female_immigrants, avg_children_per_female_mixed)


# Callback for population trend graph
@app.callback(
    Output("population-trend", "figure"),
    Input("simulation-data", "data")
)
def create_population_trend(data):
    (years, total_population, *_ ) = generate_derived_data(data)

    fig = go.Figure()
    if years:
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
    Input("simulation-data", "data")
)
def create_population_breakdown(data):
    (years, _, native_population, immigrant_1_population,
     immigrant_2_population, mixed_population, *_ ) = generate_derived_data(data)

    fig = go.Figure()
    if years:
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
    Input("simulation-data", "data")
)
def create_immigrant_percentage(data):
    (years, _, _, _, _, _, immigrant_percentage, *_ ) = generate_derived_data(data)

    fig = go.Figure()
    if years:
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


# Callback for fertility chart
@app.callback(
    Output("fertility-chart", "figure"),
    Input("simulation-data", "data")
)
def update_fertility_chart(data):
    (years, _, _, _, _, _, _, 
     avg_children_per_female_natives,
     avg_children_per_female_immigrants,
     avg_children_per_female_mixed) = generate_derived_data(data)

    fig = go.Figure()
    if years:
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


# For the age-sex pyramid and gene histogram, we need the hoverData from population-trend to determine the year.
# If no hoverData is provided, default to the last available year.

@app.callback(
    Output("age-sex-pyramid", "figure"),
    Input("simulation-data", "data"),
    Input("population-trend", "hoverData")
)
def update_age_sex_pyramid(data, hoverData):
    (years, _, _, _, _, _, _,
     _, _, _) = generate_derived_data(data)
    
    if len(years) == 0:
        # Return a blank figure or a figure with a message
        fig = go.Figure()
        fig.update_layout(title="No data yet. Advance the simulation.")
        return fig
    
    # Determine the hovered year, default to last year if none
    if hoverData and "points" in hoverData:
        year = hoverData["points"][0]["x"]
    else:
        year = years[-1]  # use last year if no hoverData

    population_data = data["population_data"][str(year)]
    if not population_data:
        fig = go.Figure()
        fig.update_layout(title="No data for the year "+str(year))
        return fig

    pyramid_data = population_data["pyramid_data"]

    simulation_batch = data["simulation_batch"]
    scaled_native_male_counts = [count * simulation_batch for count in pyramid_data["native_male_counts"]]
    scaled_immigrant_male_counts = [count * simulation_batch for count in pyramid_data["immigrant_male_counts"]]
    scaled_native_female_counts = [count * simulation_batch for count in pyramid_data["native_female_counts"]]
    scaled_immigrant_female_counts = [count * simulation_batch for count in pyramid_data["immigrant_female_counts"]]

    #scaled_male_counts = scaled_native_male_counts + scaled_immigrant_male_counts
    #scaled_female_counts = scaled_native_female_counts + scaled_immigrant_female_counts

    max_male = 250000 #max([abs(c) for c in scaled_male_counts]) if scaled_male_counts else 0
    max_female = 250000 #max([abs(c) for c in scaled_female_counts]) if scaled_female_counts else 0
    max_pyramid_count = max(max_male, max_female)

    fig = go.Figure()

    # Males
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
    # Females
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
            range=[-max_pyramid_count * 1.1, max_pyramid_count * 1.1],
        ),
        yaxis=dict(title="Age Group", autorange="reversed"),
        barmode="relative",
        bargap=0.1,
        hovermode="y unified",
        legend_title="Population Groups"
    )

    return fig


@app.callback(
    Output("immigrant-gene-histogram", "figure"),
    Input("simulation-data", "data"),
    Input("population-trend", "hoverData")
)
def update_immigrant_gene_histogram(data, hoverData):
    (years, _, _, _, _, _, _,
     _, _, _) = generate_derived_data(data)
    simulation_batch = data["simulation_batch"]
    max_hist_count = 10000000 #int(data["max_hist_count"])

    if not years:
        # No data yet
        return go.Figure()

    # Determine the hovered year
    if hoverData and "points" in hoverData:
        year = hoverData["points"][0]["x"]
    else:
        year = years[-1]

    population_data = data["population_data"].get(str(year))
    if not population_data:
        return go.Figure()

    gene_values = population_data["gene_values"]

    native_values = [gv.get('native', 0) for gv in gene_values]
    immigrant1_values = [gv.get('immigrant_1', 0) for gv in gene_values]
    immigrant2_values = [gv.get('immigrant_2', 0) for gv in gene_values]

    bins = np.arange(0, 110, 10)
    hist_native, _ = np.histogram(native_values, bins=bins)
    hist_immigrant1, _ = np.histogram(immigrant1_values, bins=bins)
    hist_immigrant2, _ = np.histogram(immigrant2_values, bins=bins)

    hist_native_scaled = hist_native * simulation_batch
    hist_immigrant1_scaled = hist_immigrant1 * simulation_batch
    hist_immigrant2_scaled = hist_immigrant2 * simulation_batch

    fig = go.Figure()
    # Bin centers
    bin_centers = bins[:-1] + 5

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist_native_scaled,
        name="Native",
        marker_color="blue",
        opacity=0.8
    ))

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist_immigrant1_scaled,
        name="Immigrant 1",
        marker_color="red",
        opacity=0.8
    ))

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist_immigrant2_scaled,
        name="Immigrant 2",
        marker_color="orange",
        opacity=0.8
    ))

    fig.update_layout(
        title=f"Immigrant Gene Value Distribution for Year {year}",
        xaxis=dict(
            title="Gene Percentage (%)",
            tickvals=bin_centers,
            ticktext=[f"{int(val)}-{int(val+10)}%" for val in bins[:-1]],
            range=[0, 100],
        ),
        yaxis=dict(
            title="Number of Individuals",
            range=[0, max_hist_count]
        ),
        barmode='group',
        hovermode="x unified",
        legend=dict(title="Gene Types")
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
