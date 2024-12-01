import random
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from tqdm import tqdm

def generate_realistic_age(sex):
    """
    Generate a random age based on a realistic age distribution.
    """
    age_groups = np.arange(101)

    weights_males = np.array([
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
    ])

    weights_females = np.array([
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
    ])

    # Normalize weights
    weights = weights_males if sex == 'male' else weights_females
    weights /= weights.sum()

    # Randomly sample an age
    return np.random.choice(age_groups, p=weights)

def prepare_age_sex_data(population):
    """Prepare age-sex pyramid data, including natives and immigrants separately."""
    age_groups = list(range(0, 101, 5))
    native_male_counts = np.zeros(len(age_groups))
    immigrant_male_counts = np.zeros(len(age_groups))
    native_female_counts = np.zeros(len(age_groups))
    immigrant_female_counts = np.zeros(len(age_groups))

    # Vectorize the data extraction
    ages = np.array([ind.age for ind in population])
    sexes = np.array([ind.sex for ind in population])
    is_immigrant = np.array([ind.is_immigrant for ind in population])

    for idx, age in enumerate(age_groups):
        age_mask = (ages >= age) & (ages < age + 5)
        native_mask = ~is_immigrant
        immigrant_mask = is_immigrant

        male_mask = sexes == 'male'
        female_mask = sexes == 'female'

        native_male_counts[idx] = -np.sum(age_mask & native_mask & male_mask)
        immigrant_male_counts[idx] = -np.sum(age_mask & immigrant_mask & male_mask)
        native_female_counts[idx] = np.sum(age_mask & native_mask & female_mask)
        immigrant_female_counts[idx] = np.sum(age_mask & immigrant_mask & female_mask)

    return {
        "age_groups": age_groups,
        "native_male_counts": native_male_counts.tolist(),
        "immigrant_male_counts": immigrant_male_counts.tolist(),
        "native_female_counts": native_female_counts.tolist(),
        "immigrant_female_counts": immigrant_female_counts.tolist()
    }

def get_death_chance(age):
    age = min(100, age)
    probability = np.array([
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
    ])

    p = probability[age] * 0.001
    return p

# Precompute immigrant age probabilities
ages_for_immigrants = np.arange(0, 100)
sigma = 10.220
immigrant_age_probabilities = np.exp(-((ages_for_immigrants - 25)**2) / (2 * sigma**2))
immigrant_age_probabilities /= immigrant_age_probabilities.sum()

class Gene:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class Individual:
    def __init__(self, id, genes, sex='male', age=0, fertility_prob=0.1, max_age=100, death_chance=0.0114):
        self.id = id
        self.genes = genes
        self.age = age
        self.sex = sex
        self.partner = None
        self.fertility_prob = fertility_prob
        self.max_age = max_age
        self.death_chance = death_chance
        self.child_count = 0

    def get_gene_value(self, gene_name):
        for gene in self.genes:
            if gene.name == gene_name:
                return gene.value
        return None

    @property
    def is_immigrant(self):
        return self.get_gene_value('immigrant_status') >= 0.5

    def age_one_year(self):
        """Age the individual by one year and check for death."""
        self.age += 1
        death_chance = get_death_chance(self.age)
        if np.random.random() < death_chance or self.age > self.max_age:
            return False
        return True

    def get_dynamic_fertility_prob(self):
        if self.is_immigrant:
            if self.child_count < 1:
                return 0.4 
            if self.child_count <= 2:
                return 0.25
            if self.child_count <= 3:
                return 0.08
            if self.child_count <= 4:
                return 0.02
            if self.child_count <= 5:
                return 0.005
            else:
                return 0.0
        else:
            if self.child_count < 1:
                return 0.3 
            if self.child_count <= 2:
                return 0.12
            if self.child_count <= 3:
                return 0.02
            if self.child_count <= 4:
                return 0.005
            if self.child_count <= 5:
                return 0.002
            else:
                return 0.0

    def have_offspring(self, next_id):
        """Attempt to have offspring, only for male-female couples."""
        if self.partner and self.sex == 'female' and np.random.random() < self.get_dynamic_fertility_prob():
            # Inherit genes from both parents
            mother_gene_value = self.get_gene_value('immigrant_status')
            father_gene_value = self.partner.get_gene_value('immigrant_status')
            offspring_gene_value = (mother_gene_value + father_gene_value) / 2
            genes = [Gene('immigrant_status', offspring_gene_value)]
            self.child_count += 1
            return Individual(next_id, genes=genes, sex=np.random.choice(['male', 'female']), fertility_prob=self.fertility_prob)
        return None

class Population:
    def __init__(self, total_population, immigrant_ratio, native_fertility, immigrant_fertility, max_age=100):
        self.population = []
        self.next_id = 1

        # Split initial population into natives and immigrants
        initial_native_count = int(total_population * (1 - immigrant_ratio))
        initial_immigrant_count = total_population - initial_native_count

        # Create initial native population with realistic ages
        for _ in range(initial_native_count):
            sex = np.random.choice(['male', 'female'])
            age = generate_realistic_age(sex)
            genes = [Gene('immigrant_status', 0.0)]  # Natives have 0.0
            self.population.append(Individual(
                self.next_id, genes=genes, sex=sex, age=age,
                fertility_prob=native_fertility, max_age=max_age
            ))
            self.next_id += 1

        # Create initial immigrant population with realistic ages
        for _ in range(initial_immigrant_count):
            sex = np.random.choice(['male', 'female'])
            age = generate_realistic_age(sex)
            genes = [Gene('immigrant_status', 1.0)]  # Immigrants have 1.0
            self.population.append(Individual(
                self.next_id, genes=genes, sex=sex, age=age,
                fertility_prob=immigrant_fertility, max_age=max_age
            ))
            self.next_id += 1

    def simulate_year(self, net_migration, max_age=100):
        """Simulate a year of aging, reproduction, and death."""
        new_population = []

        # Age individuals and remove those who die
        for individual in self.population:
            if individual.age_one_year():
                new_population.append(individual)

        potential_partners = [
            ind for ind in new_population
            if ind.partner is None
            and ind.sex == 'female'
            and 18 <= ind.age <= 40
        ]

        # Reset partners
        for individual in new_population:
            individual.partner = None

        # Prepare lists of partnerable individuals
        partnerable_males = [ind for ind in new_population if ind.sex == 'male' and ind.partner is None and 18 <= ind.age <= 70]
        partnerable_females = [ind for ind in new_population if ind.sex == 'female' and ind.partner is None and 18 <= ind.age <= 40]

        # Shuffle the lists
        np.random.shuffle(partnerable_males)
        np.random.shuffle(partnerable_females)

        # Pair up individuals
        min_len = min(len(partnerable_males), len(partnerable_females))
        for i in range(min_len):
            male = partnerable_males[i]
            female = partnerable_females[i]
            if abs(male.age - female.age) <= 5:
                male.partner = female
                female.partner = male

        # Simulate having offspring
        offspring_list = []
        for individual in new_population:
            offspring = individual.have_offspring(self.next_id)
            if offspring:
                offspring_list.append(offspring)
                self.next_id += 1

        new_population.extend(offspring_list)

        # Add net migration
        for _ in range(int(net_migration)):
            age = np.random.choice(ages_for_immigrants, p=immigrant_age_probabilities)
            genes = [Gene('immigrant_status', 1.0)]  # New immigrants have 1.0
            new_population.append(Individual(
                self.next_id, genes=genes, age=age,
                sex=np.random.choice(['male', 'female']), fertility_prob=0.017
            ))
            self.next_id += 1

        self.population = new_population

    def get_population_statistics(self):
        """Calculate population statistics, including sex distribution."""
        total = len(self.population)
        sexes = np.array([ind.sex for ind in self.population])
        gene_values = np.array([ind.get_gene_value('immigrant_status') for ind in self.population])

        males = np.sum(sexes == 'male')
        females = total - males
        natives = np.sum(gene_values == 0.0)
        immigrants = np.sum(gene_values == 1.0)
        mixed = total - natives - immigrants
        return {
            "total_population": total,
            "native_population": natives,
            "immigrant_population": immigrants,
            "mixed_population": mixed,
            "male_population": males,
            "female_population": females,
            "immigrant_percentage": ((immigrants + mixed) / total) * 100
        }

population_data = {}

# Simulation with a starting population of 5.6 million
def run_large_simulation(years=100, net_migration=56.0):
    total_population = 5600  # Adjusted for simulation scale
    immigrant_ratio = 0.062
    native_fertility = 0.0126
    immigrant_fertility = 0.017

    pop = Population(total_population, immigrant_ratio, native_fertility, immigrant_fertility)
    stats = []

    max_count = 0
    max_hist_count = 0

    # Use tqdm for progress bar
    for year in tqdm(range(years), desc="Simulating years"):
        pop.simulate_year(net_migration)
        stat = pop.get_population_statistics()
        stats.append(stat)

        # Prepare data for plotting
        pyramid_data = prepare_age_sex_data(pop.population)
        gene_values = [ind.get_gene_value('immigrant_status') for ind in pop.population]

        # Compute counts for max_count
        counts = (
            pyramid_data['native_male_counts'] + pyramid_data['immigrant_male_counts'] +
            pyramid_data['native_female_counts'] + pyramid_data['immigrant_female_counts']
        )
        counts_abs = [abs(count) for count in counts]
        if counts_abs:
            max_count = max(max_count, max(counts_abs), max_count)

        # Compute histogram counts for max_hist_count
        bins = np.arange(0, 1.1, 0.1)  # Bins from 0 to 1 in steps of 0.1
        hist, _ = np.histogram(gene_values, bins=bins)
        max_hist_count = max(max_hist_count, max(hist), max_hist_count)

        # Store aggregated data
        population_data[year] = {
            "pyramid_data": pyramid_data,
            "gene_values": gene_values
        }

    # Add buffer to max_count and max_hist_count
    max_count = int(max_count * 1.1)
    max_hist_count = int(max_hist_count * 1.1)

    return stats, max_count, max_hist_count

# Run the simulation once
stats, max_count, max_hist_count = run_large_simulation()

# Prepare data for plotting
years = list(range(len(stats)))
total_population = [stat['total_population'] for stat in stats]
native_population = [stat['native_population'] for stat in stats]
immigrant_population = [stat['immigrant_population'] for stat in stats]
mixed_population = [stat['mixed_population'] for stat in stats]
immigrant_percentage = [stat['immigrant_percentage'] for stat in stats]

# Dash app initialization
app = dash.Dash(__name__)

# Updated Layout
app.layout = html.Div([
    # Top row with two smaller charts side by side
    html.Div([
        html.Div([
            dcc.Graph(
                id="population-breakdown",
                config={"displayModeBar": False},
                style={"height": "300px"}
            )
        ], style={"width": "48%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(
                id="immigrant-percentage",
                config={"displayModeBar": False},
                style={"height": "300px"}
            )
        ], style={"width": "48%", "display": "inline-block"})
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

# Callback for age-sex pyramid
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

    fig = go.Figure()

    # Plot native males
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=pyramid_data["native_male_counts"],
        orientation="h",
        name="Native Males",
        marker_color="blue"
    ))
    # Plot immigrant males
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=pyramid_data["immigrant_male_counts"],
        orientation="h",
        name="Immigrant Males",
        marker_color="lightblue"
    ))
    # Plot native females
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=pyramid_data["native_female_counts"],
        orientation="h",
        name="Native Females",
        marker_color="red"
    ))
    # Plot immigrant females
    fig.add_trace(go.Bar(
        y=pyramid_data["age_groups"],
        x=pyramid_data["immigrant_female_counts"],
        orientation="h",
        name="Immigrant Females",
        marker_color="lightsalmon"
    ))

    fig.update_layout(
        title=f"Age-Sex Pyramid for Year {year}",
        xaxis=dict(
            title="Population",
            range=[-max_count, max_count]  # Set static x-axis range
        ),
        yaxis=dict(title="Age Group", autorange="reversed"),
        barmode="relative",
        hovermode="y unified"
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

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
