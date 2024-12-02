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
    def __init__(self, id, genes, sex='male', age=0, fertility=2.1, max_age=100, death_chance=0.0114):
        self.id = id
        self.genes = genes
        self.age = age
        self.sex = sex
        self.partner = None
        self.fertility = fertility
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
            self.partner.child_count += 1  # Also increment the partner's child count
            return Individual(next_id, genes=genes, sex=np.random.choice(['male', 'female']), fertility=self.fertility)
        return None

class Population:
    def __init__(self, total_population, immigrant_ratio, native_fertility, immigrant_fertility, max_age=100):
        self.population = []
        self.next_id = 1

        # Lists to store deceased females' child counts and immigration status
        self.deceased_females_natives = []
        self.deceased_females_immigrants = []
        self.deceased_females_mixed = []

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
                fertility=native_fertility, max_age=max_age
            ))
            self.next_id += 1

        # Create initial immigrant population with realistic ages
        for _ in range(initial_immigrant_count):
            sex = np.random.choice(['male', 'female'])
            age = generate_realistic_age(sex)
            genes = [Gene('immigrant_status', 1.0)]  # Immigrants have 1.0
            self.population.append(Individual(
                self.next_id, genes=genes, sex=sex, age=age,
                fertility=immigrant_fertility, max_age=max_age
            ))
            self.next_id += 1

    def simulate_year(self, net_migration, max_age=100):
        """Simulate a year of aging, reproduction, and death."""
        new_population = []

        # Age individuals and remove those who die
        for individual in self.population:
            alive = individual.age_one_year()
            if alive:
                new_population.append(individual)
            else:
                # If the individual is a female, store her child count and immigration status
                if individual.sex == 'female':
                    gene_value = individual.get_gene_value('immigrant_status')
                    if gene_value == 0.0:
                        self.deceased_females_natives.append(individual.child_count)
                    elif gene_value == 1.0:
                        self.deceased_females_immigrants.append(individual.child_count)
                    else:
                        self.deceased_females_mixed.append(individual.child_count)

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
                sex=np.random.choice(['male', 'female']), fertility=1.7
            ))
            self.next_id += 1

        self.population = new_population

        # Calculate average child counts for deceased females
        self.avg_children_native = np.mean(self.deceased_females_natives) if self.deceased_females_natives else 0
        self.avg_children_immigrant = np.mean(self.deceased_females_immigrants) if self.deceased_females_immigrants else 0
        self.avg_children_mixed = np.mean(self.deceased_females_mixed) if self.deceased_females_mixed else 0

    def create_realistic_child_count(self):
        """Adjust child counts to ensure realistic averages for natives and immigrants."""

        def compute_child_count_average(females, target_avg):
            """
            Compute the required extra children to reach the target average.
            Returns the number of extra children needed.
            """
            current_total = sum(ind.child_count for ind in females)
            required_total = int(len(females) * target_avg)
            extra_children_needed = required_total - current_total
            return extra_children_needed

        def distribute_extra_children(females, target_avg):
            # Continue adding children until the average is fulfilled
            while True:
                extra_children_needed = compute_child_count_average(females, target_avg)
                if extra_children_needed <= 0:
                    break

                # Shuffle the list for fairness
                random.shuffle(females)

                # Add children incrementally
                for i in range(extra_children_needed):
                    females[i % len(females)].child_count += 1

        # Filter native and immigrant females
        native_females = [ind for ind in self.population if ind.sex == 'female' and ind.get_gene_value('immigrant_status') == 0.0 and 18 <= ind.age]
        immigrant_females = [ind for ind in self.population if ind.sex == 'female' and ind.get_gene_value('immigrant_status') == 1.0 and 18 <= ind.age]

        # Adjust child counts
        distribute_extra_children(native_females, 1.26)
        distribute_extra_children(immigrant_females, 1.7)

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
            "immigrant_percentage": ((immigrants + mixed) / total) * 100,
            "avg_children_native": self.avg_children_native,
            "avg_children_immigrant": self.avg_children_immigrant,
            "avg_children_mixed": self.avg_children_mixed
        }

population_data = {}

# Simulation with a starting population of 5.6 million
def run_large_simulation(years=100, net_migration=560.0):
    total_population = 56000  # Adjusted for simulation scale
    immigrant_ratio = 0.062
    native_fertility = 1.26
    immigrant_fertility = 1.7

    pop = Population(total_population, immigrant_ratio, native_fertility, immigrant_fertility)
    pop.create_realistic_child_count()
    stats = []

    avg_children_per_female_natives = []
    avg_children_per_female_immigrants = []
    avg_children_per_female_mixed = []

    max_count = 0
    max_hist_count = 0

    # Use tqdm for progress bar
    for year in tqdm(range(years), desc="Simulating years"):
        pop.simulate_year(net_migration)
        stat = pop.get_population_statistics()
        stats.append(stat)

        # Collect average child counts
        avg_children_per_female_natives.append(stat['avg_children_native'])
        avg_children_per_female_immigrants.append(stat['avg_children_immigrant'])
        avg_children_per_female_mixed.append(stat['avg_children_mixed'])

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

    return stats, max_count, max_hist_count, avg_children_per_female_natives, avg_children_per_female_immigrants, avg_children_per_female_mixed, population_data

