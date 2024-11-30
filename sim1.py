import random
import matplotlib.pyplot as plt
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import copy

def generate_realistic_age(sex):
    """
    Generate a random age based on a realistic age distribution.
    """
    # Define age groups (0 to 100)
    age_groups = list(range(101))

    weights_males = [
        0.021, 0.021, 0.021, 0.021, 0.021,
        0.025, 0.025, 0.025, 0.025, 0.025,
        0.029, 0.029, 0.029, 0.029, 0.029,
        0.029, 0.029, 0.029, 0.029, 0.029,
        0.028, 0.028, 0.028, 0.028, 0.028,
        0.030, 0.030, 0.030, 0.030, 0.030,
        0.034, 0.034, 0.034, 0.034, 0.034,
        0.033, 0.033, 0.033, 0.033, 0.033,
        0.034, 0.034, 0.034, 0.034, 0.034,
        0.032, 0.032, 0.032, 0.032, 0.032,
        0.029, 0.029, 0.029, 0.029, 0.029,
        0.032, 0.032, 0.032, 0.032, 0.032,
        0.032, 0.032, 0.032, 0.032, 0.032,
        0.030, 0.030, 0.030, 0.030, 0.030,
        0.028, 0.028, 0.028, 0.028, 0.028,
        0.025, 0.025, 0.025, 0.025, 0.025,
        0.013, 0.013, 0.013, 0.013, 0.013,
        0.007, 0.007, 0.007, 0.007, 0.007,
        0.002, 0.002, 0.002, 0.002, 0.002,
        0.000, 0.000, 0.000, 0.000, 0.000,
        0.000
    ]

    weights_females = [
        0.020, 0.020, 0.020, 0.020, 0.020, 
        0.024, 0.024, 0.024, 0.024, 0.024, 
        0.028, 0.028, 0.028, 0.028, 0.028, 
        0.028, 0.028, 0.028, 0.028, 0.028, 
        0.027, 0.027, 0.027, 0.027, 0.027, 
        0.029, 0.029, 0.029, 0.029, 0.029, 
        0.032, 0.032, 0.032, 0.032, 0.032, 
        0.031, 0.031, 0.031, 0.031, 0.031, 
        0.032, 0.032, 0.032, 0.032, 0.032, 
        0.030, 0.030, 0.030, 0.030, 0.030, 
        0.028, 0.028, 0.028, 0.028, 0.028, 
        0.032, 0.032, 0.032, 0.032, 0.032, 
        0.033, 0.033, 0.033, 0.033, 0.033, 
        0.032, 0.032, 0.032, 0.032, 0.032, 
        0.032, 0.032, 0.032, 0.032, 0.032, 
        0.030, 0.030, 0.030, 0.030, 0.030, 
        0.018, 0.018, 0.018, 0.018, 0.018, 
        0.012, 0.012, 0.012, 0.012, 0.012, 
        0.006, 0.006, 0.006, 0.006, 0.006, 
        0.002, 0.002, 0.002, 0.002, 0.002, 
        0.000
    ]

    # Normalize weights
    if sex == 'male':
        weights = np.array(weights_males)
    else:
        weights = np.array(weights_females)
    weights /= weights.sum()

    # Randomly sample an age
    return np.random.choice(age_groups, p=weights)

# Helper function to prepare age-sex pyramid data for a given year
def prepare_age_sex_data(population):
    age_groups = list(range(0, 101, 5))
    male_counts = [-sum(1 for ind in population if ind.sex == 'male' and age <= ind.age < age+5) for age in age_groups]
    female_counts = [sum(1 for ind in population if ind.sex == 'female' and age <= ind.age < age+5) for age in age_groups]
    return {"age_groups": age_groups, "male_counts": male_counts, "female_counts": female_counts}

    # Updated interactive plotting function
def plot_interactive_population(population_data):
    # Extract total population over time
    years = list(population_data.keys())
    total_population = [len(data['population']) for data in population_data.values()]

    # Line chart for total population on a secondary y-axis
    line_fig = go.Scatter(
        x=years,
        y=total_population,
        mode="lines+markers",
        name="Total Population",
        line=dict(color="green", width=3),
        yaxis="y2",  # Assign to secondary y-axis
    )

    # Create age-sex pyramid bars for each year
    bar_figs = []
    for year, data in population_data.items():
        pyramid_data = prepare_age_sex_data(data["population"])
        visible = (year == 0)  # Show only the first year's bars initially

        # Add male and female bars
        bar_figs.append(go.Bar(
            y=pyramid_data["age_groups"],
            x=pyramid_data["male_counts"],
            orientation="h",
            name=f"Males (Year {year})",
            marker_color="blue",
            visible=visible,
        ))
        bar_figs.append(go.Bar(
            y=pyramid_data["age_groups"],
            x=pyramid_data["female_counts"],
            orientation="h",
            name=f"Females (Year {year})",
            marker_color="pink",
            visible=visible,
        ))

    # Create dropdown menu for interactivity
    buttons = []
    for year in years:
        visibility = [False] * len(years) * 2  # Set all bars to invisible
        visibility[year * 2] = True  # Male bars for the current year
        visibility[year * 2 + 1] = True  # Female bars for the current year
        visibility += [True] * len(years)  # Line graph always visible
        buttons.append(dict(
            label=f"Year {year}",
            method="update",
            args=[{"visible": visibility}]
        ))

    # Combine figures
    fig = go.Figure()
    fig.add_traces(bar_figs)
    fig.add_trace(line_fig)

    fig.update_layout(
        title="Population Dynamics with Age-Sex Pyramid",
        xaxis=dict(title="Population"),
        yaxis=dict(title="Age Group", autorange="reversed"),  # For age-sex pyramid
        yaxis2=dict(
            title="Total Population",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        barmode="relative",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
        )]
    )

    fig.show()
    fig.write_html("age_sex_pyramid.html")

def get_death_chance(age):
    age = min(100, age)
    probability = [
        1.75, 0.17, 0.18, 0.02, 0.19,
        0.08, 0.02, 0.09, 0.18, 0.11,
        0.03, 0.08, 0.05, 0.11, 0.16,
        0.16, 0.27, 0.29, 0.61, 0.70,
        0.71, 0.78, 0.56, 0.64, 0.78,
        0.44, 0.56, 0.52, 0.55, 0.66,
        0.56, 0.49, 0.69, 0.73, 0.85,
        0.59, 0.69, 0.94, 0.71, 1.28,
        0.99, 1.17, 1.08, 1.31, 1.62,
        1.42, 1.53, 1.64, 1.77, 2.06,
        2.11, 2.73, 3.22, 3.20, 3.34,
        3.78, 4.33, 4.66, 4.74, 5.26,
        6.39, 6.59, 7.55, 7.79, 8.42,
        9.80, 11.13, 11.95, 12.61, 13.31,
        14.85, 16.68, 18.07, 21.20, 22.67,
        25.64, 27.92, 33.43, 32.57, 39.00,
        44.47, 48.75, 52.88, 65.98, 70.08,
        80.80, 88.90, 102.03, 117.78, 131.23,
        149.01, 166.65, 189.42, 211.54, 242.88,
        269.17, 280.13, 299.23, 321.22, 379.78,
        1000.0
    ]

    p = probability[age] * 0.001
    return p

import pandas as pd

def plot_age_sex_pyramid(population):
    """Plot the age-sex pyramid of the population."""
    # Create age-sex groups
    age_groups = list(range(0, 101, 5))
    male_counts = [-sum(1 for ind in population if ind.sex == 'male' and age <= ind.age < age+5) for age in age_groups]
    female_counts = [sum(1 for ind in population if ind.sex == 'female' and age <= ind.age < age+5) for age in age_groups]

    # Plot the pyramid
    plt.figure(figsize=(10, 6))
    plt.barh(age_groups, male_counts, color='blue', label='Males')
    plt.barh(age_groups, female_counts, color='pink', label='Females')
    plt.xlabel("Population Count")
    plt.ylabel("Age Group")
    plt.title("Age-Sex Pyramid")
    plt.legend()
    plt.grid()
    plt.show()

class Individual:
    def __init__(self, id, is_immigrant, sex='male', age=0, fertility_prob=0.1, max_age=100, death_chance=0.0114):
        self.id = id
        self.is_immigrant = is_immigrant
        self.age = age
        self.sex = sex #if sex else random.choice(['male', 'female'])  # Randomly assign sex if not provided
        self.partner = None
        self.fertility_prob = fertility_prob
        self.max_age = max_age
        self.death_chance = death_chance

    def age_one_year(self):
        """Age the individual by one year and check for death."""
        self.age += 1
        death_chance = get_death_chance(self.age)
        if np.random.random() < death_chance or self.age > self.max_age:
            return False
        return True

    def find_partner(self, population):
        """Attempt to find a partner of the opposite sex."""
        if self.partner is not None or not (18 <= self.age <= 40):
            return  # Skip if already partnered or outside partnerable age range
        
        potential_partners = [
            ind for ind in population
            if ind.partner is None
            and ind is not self
            and ind.sex != self.sex  # Opposite sex
            and 18 <= ind.age <= 40
            and abs(ind.age - self.age) <= 5
        ]
        if potential_partners:
            self.partner = np.random.choice(potential_partners)
            self.partner.partner = self
    
    def have_offspring(self, next_id):
        """Attempt to have offspring, only for male-female couples."""
        if self.partner and self.sex == 'female' and np.random.random() < self.fertility_prob:
            return Individual(next_id, self.is_immigrant, sex=np.random.choice(['male', 'female']), fertility_prob=self.fertility_prob)
        return None

class Population:
    def __init__(self, total_population, immigrant_ratio, native_fertility, immigrant_fertility, max_age=100):
        self.population = [None] * total_population
        self.next_id = 1

        # Split initial population into natives and immigrants
        initial_native_count = int(total_population * (1 - immigrant_ratio))
        initial_immigrant_count = total_population - initial_native_count

        # Create initial population with realistic ages
        for _ in range(initial_native_count):
            print(str(self.next_id))
            sex = np.random.choice(['male', 'female'])
            age = generate_realistic_age(sex)
            self.population[self.next_id-1] = Individual(
                self.next_id, is_immigrant=False, sex=sex, age=age,
                fertility_prob=native_fertility, max_age=max_age
            )
            self.next_id += 1
        
        for _ in range(initial_immigrant_count):
            sex = np.random.choice(['male', 'female'])
            age = generate_realistic_age(sex)
            self.population[self.next_id-1] = Individual(
                self.next_id, is_immigrant=True, sex=sex, age=age,
                fertility_prob=immigrant_fertility, max_age=max_age
            )
            self.next_id += 1

    def simulate_year(self, net_migration, max_age=100):
        """Simulate a year of aging, reproduction, and death."""
        new_population = []

        # Age individuals and remove those who die
        for individual in self.population:
            if individual.age_one_year():
                new_population.append(individual)

        # Simulate finding partners
        for individual in new_population:
            individual.find_partner(new_population)

        # Simulate having offspring
        for individual in new_population:
            offspring = individual.have_offspring(self.next_id)
            if offspring:
                new_population.append(offspring)
                self.next_id += 1

        # Add net migration
        for _ in range(int(net_migration)):
            age = np.random.randint(0, max_age)
            new_population.append(Individual(
                self.next_id, is_immigrant=True, age=age,
                sex=np.random.choice(['male', 'female']), fertility_prob=0.017
            ))
            self.next_id += 1

        self.population = new_population

    def get_population_statistics(self):
        """Calculate population statistics, including sex distribution."""
        total = len(self.population)
        males = sum(1 for ind in self.population if ind.sex == 'male')
        females = total - males
        natives = sum(1 for ind in self.population if not ind.is_immigrant)
        immigrants = total - natives
        return {
            "total_population": total,
            "native_population": natives,
            "immigrant_population": immigrants,
            "male_population": males,
            "female_population": females,
            "immigrant_percentage": (immigrants / total) * 100
        }

population_data = {}

# Simulation with a starting population of 5.6 million
def run_large_simulation(years=100, net_migration=56.0):
    total_population = 5600  # 5.6 million
    immigrant_ratio = 0.062
    native_fertility = 0.0126
    immigrant_fertility = 0.017

    pop = Population(total_population, immigrant_ratio, native_fertility, immigrant_fertility)
    stats = []

    for year in range(years):
        pop.simulate_year(net_migration)
        stats.append(pop.get_population_statistics())
        print("year "+str(year))
        population_data[year] = {"population": copy.deepcopy(pop.population)}
        
 #   plot_age_sex_pyramid(pop.population)

    return stats

# Run the simulation
stats = run_large_simulation()

# Plotting total, native, and immigrant populations
years = list(range(len(stats)))
total_population = [stat['total_population'] for stat in stats]
native_population = [stat['native_population'] for stat in stats]
immigrant_population = [stat['immigrant_population'] for stat in stats]
immigrant_percentage = [stat['immigrant_percentage'] for stat in stats]

# Plot total, native, and immigrant populations
plt.figure(figsize=(12, 6))
plt.plot(years, total_population, label="Total Population", linewidth=2)
plt.plot(years, native_population, label="Native Population", linestyle='--')
plt.plot(years, immigrant_population, label="Immigrant Population", linestyle=':')
plt.title("Population Dynamics Over Time (Random Ages)")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.show()

# Plot immigrant percentage over time
plt.figure(figsize=(12, 6))
plt.plot(years, immigrant_percentage, label="Immigrant Percentage", color='orange', linewidth=2)
plt.title("Immigrant Percentage Over Time (Random Ages)")
plt.xlabel("Year")
plt.ylabel("Immigrant Percentage (%)")
plt.legend()
plt.grid()
plt.show()

#plot_interactive_population(population_data)

# Dash app initialization
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
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
])

# Callback for total population trend graph
@app.callback(
    Output("population-trend", "figure"),
    Input("population-trend", "hoverData")  # Correct Input dependency
)
def create_population_trend(_):
    # Total population trend
    years = list(population_data.keys())
    total_population = [len(data["population"]) for data in population_data.values()]

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

# Callback for age-sex pyramid
@app.callback(
    Output("age-sex-pyramid", "figure"),
    Input("population-trend", "hoverData")  # Fix to connect with the correct hoverData
)
def update_age_sex_pyramid(hoverData):
    # Determine the hovered year
    if hoverData and "points" in hoverData:
        year = hoverData["points"][0]["x"]
    else:
        year = 0  # Default to the first year if no hover data is available

    population = population_data[year]["population"]
    pyramid_data = prepare_age_sex_data(population)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=pyramid_data["male_counts"],
        orientation="h",
        name="Males",
        marker_color="blue"
    ))
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=pyramid_data["female_counts"],
        orientation="h",
        name="Females",
        marker_color="pink"
    ))

    fig.update_layout(
        title=f"Age-Sex Pyramid for Year {year}",
        xaxis=dict(title="Population"),
        yaxis=dict(title="Age Group", autorange="reversed"),
        barmode="relative",
        hovermode="y unified"
    )

    return fig

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
