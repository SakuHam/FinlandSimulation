import random
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

# Define integer constants for keys
AGE_GROUPS_KEY = 0
NATIVE_MALE_COUNTS_KEY = 1
IMMIGRANT_MALE_COUNTS_KEY = 2
NATIVE_FEMALE_COUNTS_KEY = 3
IMMIGRANT_FEMALE_COUNTS_KEY = 4

# Define a mapping from key constants back to their string names
KEY_TO_STRING = {
    AGE_GROUPS_KEY: "age_groups",
    NATIVE_MALE_COUNTS_KEY: "native_male_counts",
    IMMIGRANT_MALE_COUNTS_KEY: "immigrant_male_counts",
    NATIVE_FEMALE_COUNTS_KEY: "native_female_counts",
    IMMIGRANT_FEMALE_COUNTS_KEY: "immigrant_female_counts"
}

# Simulation parameters
years = 100
simulation_batch = 1000
net_migration = int(56000 / simulation_batch)
total_population = int(5600000 / simulation_batch)
immigrant_ratio = 0.062
native_fertility = 1.26
immigrant_fertility = 1.7

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

    weights = weights_males if sex == 'male' else weights_females
    weights /= weights.sum()
    return np.random.choice(age_groups, p=weights)

def prepare_age_sex_data(population):
    """Prepare age-sex pyramid data, including natives and immigrants, using integer keys."""
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
        AGE_GROUPS_KEY: age_groups,
        NATIVE_MALE_COUNTS_KEY: native_male_counts.tolist(),
        IMMIGRANT_MALE_COUNTS_KEY: immigrant_male_counts.tolist(),
        NATIVE_FEMALE_COUNTS_KEY: native_female_counts.tolist(),
        IMMIGRANT_FEMALE_COUNTS_KEY: immigrant_female_counts.tolist()
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

class GeneGroup:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = {}
        else:
            self.genes = genes  # Should be a dict mapping from gene name to gene value

    def add_gene(self, name, value):
        self.genes[name] = value

    def get_gene_value(self, name):
        return self.genes.get(name, 0.0)  # Default to 0.0 if gene not present

    def calculate_percentages(self):
        total = sum(self.genes.values())
        percentages = {name: (value / total) * 100 if total != 0 else 0 for name, value in self.genes.items()}
        return percentages

class Individual:
    def __init__(self, id, gene_group, sex='male', age=0, fertility=2.1, max_age=100, death_chance=0.0114):
        self.id = id
        self.gene_group = gene_group
        self.age = age
        self.sex = sex
        self.partner = None
        self.fertility = fertility
        self.max_age = max_age
        self.death_chance = death_chance
        self.child_count = 0

    def get_gene_value(self, gene_name):
        return self.gene_group.get_gene_value(gene_name)

    @property
    def is_immigrant(self):
        # Consider an individual an immigrant if 'native' gene is less than 0.5
        return self.get_gene_value('native') < 0.5

    @property
    def immigrant_type(self):
        # Determine the immigrant type based on the highest immigrant gene value
        imm1 = self.get_gene_value('immigrant_1')
        imm2 = self.get_gene_value('immigrant_2')
        if imm1 >= imm2:
            return 'immigrant_1'
        elif imm2 > imm1:
            return 'immigrant_2'
        elif imm1 == imm2:
            return 'mixed'
        else:
            return "native"  # Native

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
            mother_genes = self.gene_group.genes
            father_genes = self.partner.gene_group.genes
            offspring_genes = {}
            for gene_name in set(mother_genes.keys()).union(father_genes.keys()):
                mother_value = mother_genes.get(gene_name, 0.0)
                father_value = father_genes.get(gene_name, 0.0)
                offspring_genes[gene_name] = (mother_value + father_value) / 2

            # Validate that the sum of gene values is approximately 1.0
            total_gene_value = sum(offspring_genes.values())
            if not np.isclose(total_gene_value, 1.0, atol=1e-2):
                raise ValueError(
                    f"Offspring gene values sum to {total_gene_value}, which is not near 1.0. "
                    f"Genes: {offspring_genes}"
                )

            gene_group = GeneGroup(genes=offspring_genes)
            self.child_count += 1
            self.partner.child_count += 1  # Also increment the partner's child count
            return Individual(next_id, gene_group=gene_group, sex=np.random.choice(['male', 'female']), fertility=self.fertility)
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

        # Create initial native population
        for _ in range(initial_native_count):
            sex = np.random.choice(['male', 'female'])
            age = generate_realistic_age(sex)
            gene_group = GeneGroup()
            gene_group.add_gene('native', 1.0)  # Natives have 'native' gene set to 1.0
            gene_group.add_gene('immigrant_1', 0.0)
            gene_group.add_gene('immigrant_2', 0.0)
            self.population.append(Individual(
                self.next_id, gene_group=gene_group, sex=sex, age=age,
                fertility=native_fertility, max_age=max_age
            ))
            self.next_id += 1

        # Create initial immigrant population
        for _ in range(initial_immigrant_count):
            sex = np.random.choice(['male', 'female'])
            age = generate_realistic_age(sex)
            gene_group = GeneGroup()
            gene_group.add_gene('native', 0.0)  # Immigrants have 'native' gene set to 0.0
            # Assign 'immigrant_1' or 'immigrant_2' with probabilities 20% and 80%
            immigrant_gene = np.random.choice(['immigrant_1', 'immigrant_2'], p=[0.2, 0.8])
            gene_group.add_gene(immigrant_gene, 1.0)
            other_immigrant_gene = 'immigrant_1' if immigrant_gene == 'immigrant_2' else 'immigrant_2'
            gene_group.add_gene(other_immigrant_gene, 0.0)
            self.population.append(Individual(
                self.next_id, gene_group=gene_group, sex=sex, age=age,
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
                # If the individual is female, store her child count based on immigrant/native status
                if individual.sex == 'female':
                    if individual.get_gene_value('native') == 1.0:
                        self.deceased_females_natives.append(individual.child_count)
                    elif individual.get_gene_value('native') == 0.0 and (
                        individual.get_gene_value('immigrant_1') == 1.0 or individual.get_gene_value('immigrant_2') == 1.0):
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
            gene_group = GeneGroup()
            gene_group.add_gene('native', 0.0)  # New immigrants have 'native' gene set to 0.0
            # Assign 'immigrant_1' or 'immigrant_2' with probabilities 20% and 80%
            immigrant_gene = np.random.choice(['immigrant_1', 'immigrant_2'], p=[0.2, 0.8])
            gene_group.add_gene(immigrant_gene, 1.0)
            other_immigrant_gene = 'immigrant_1' if immigrant_gene == 'immigrant_2' else 'immigrant_2'
            gene_group.add_gene(other_immigrant_gene, 0.0)
            new_population.append(Individual(
                self.next_id, gene_group=gene_group, age=age,
                sex=np.random.choice(['male', 'female']), fertility=immigrant_fertility
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

                random.shuffle(females)

                for i in range(extra_children_needed):
                    females[i % len(females)].child_count += 1

        native_females = [ind for ind in self.population if ind.sex == 'female' and ind.get_gene_value('native') == 1.0 and 18 <= ind.age]
        immigrant_females = [ind for ind in self.population if ind.sex == 'female' and ind.get_gene_value('native') == 0.0 and 18 <= ind.age]

        distribute_extra_children(native_females, native_fertility)
        distribute_extra_children(immigrant_females, immigrant_fertility)

    def get_population_statistics(self):
        """Calculate population statistics, including sex distribution."""
        total = len(self.population)
        sexes = np.array([ind.sex for ind in self.population])

        natives = sum(ind.get_gene_value('native') == 1.0 for ind in self.population)
        immigrants_1 = sum(ind.get_gene_value('immigrant_1') == 1.0 for ind in self.population)
        immigrants_2 = sum(ind.get_gene_value('immigrant_2') == 1.0 for ind in self.population)
        mixed = total - natives - immigrants_1 - immigrants_2

        males = np.sum(sexes == 'male')
        females = total - males

        immigrant_percentage = ((immigrants_1 + immigrants_2 + mixed) / total) * 100

        return {
            "total_population": total,
            "native_population": natives,
            "immigrant_1_population": immigrants_1,
            "immigrant_2_population": immigrants_2,
            "mixed_population": mixed,
            "male_population": males,
            "female_population": females,
            "immigrant_percentage": immigrant_percentage,
            "avg_children_native": self.avg_children_native,
            "avg_children_immigrant": self.avg_children_immigrant,
            "avg_children_mixed": self.avg_children_mixed
        }

population_data = {}

def run_large_simulation(net_migration):
    pop = Population(total_population, immigrant_ratio, native_fertility, immigrant_fertility)
    pop.create_realistic_child_count()
    stats = []

    avg_children_per_female_natives = []
    avg_children_per_female_immigrants = []
    avg_children_per_female_mixed = []

    max_count = 0
    max_hist_count = 0

    for year in tqdm(range(years), desc=f"Simulating years with net_migration={net_migration * simulation_batch}"):
        pop.simulate_year(net_migration)
        stat = pop.get_population_statistics()
        stats.append(stat)

        avg_children_per_female_natives.append(stat['avg_children_native'])
        avg_children_per_female_immigrants.append(stat['avg_children_immigrant'])
        avg_children_per_female_mixed.append(stat['avg_children_mixed'])

        # Prepare data for plotting
        pyramid_data = prepare_age_sex_data(pop.population)
        gene_values = [ind.gene_group.calculate_percentages() for ind in pop.population]

        counts = (
            pyramid_data[NATIVE_MALE_COUNTS_KEY] + pyramid_data[IMMIGRANT_MALE_COUNTS_KEY] +
            pyramid_data[NATIVE_FEMALE_COUNTS_KEY] + pyramid_data[IMMIGRANT_FEMALE_COUNTS_KEY]
        )
        counts_abs = [abs(count) for count in counts]
        if counts_abs:
            max_count = max(max_count, max(counts_abs), max_count)

        immigrant_percentages = [sum([gv.get('immigrant_1', 0), gv.get('immigrant_2', 0)]) for gv in gene_values]
        bins = np.arange(0, 101, 10) 
        hist, _ = np.histogram(immigrant_percentages, bins=bins)
        max_hist_count = max(max_hist_count, max(hist), max_hist_count)

        population_data[year] = {
            "pyramid_data": pyramid_data,
            "gene_values": gene_values
        }

    max_count = int(max_count * 1.1)
    max_hist_count = int(max_hist_count * 1.1)

    return stats, max_count, max_hist_count, avg_children_per_female_natives, avg_children_per_female_immigrants, avg_children_per_female_mixed, population_data, simulation_batch

def monte_carlo_simulations(num_simulations, net_migration_values=[56000, 40000, 25000]):
    """
    Run multiple sets of Monte Carlo simulations for different net_migration values.
    Returns a dictionary keyed by the net_migration scenario with the results.
    """
    results_by_immigration = {}

    for nm_val in net_migration_values:
        nm_per_batch = int(nm_val / simulation_batch)

        results = Parallel(n_jobs=10)(
            delayed(run_large_simulation)(nm_per_batch)
            for _ in range(num_simulations)
        )

        all_stats = [r[0] for r in results]
        all_max_count = [r[1] for r in results]
        all_max_hist_count = [r[2] for r in results]
        all_avg_natives = [r[3] for r in results]
        all_avg_immigrants = [r[4] for r in results]
        all_avg_mixed = [r[5] for r in results]
        all_population_data = [r[6] for r in results]
        all_simulation_batch = [r[7] for r in results]

        avg_max_count = np.mean(all_max_count)
        avg_max_hist_count = np.mean(all_max_hist_count)
        avg_simulation_batch = int(np.mean(all_simulation_batch))

        avg_children_per_female_natives = np.mean(all_avg_natives, axis=0).tolist()
        avg_children_per_female_immigrants = np.mean(all_avg_immigrants, axis=0).tolist()
        avg_children_per_female_mixed = np.mean(all_avg_mixed, axis=0).tolist()

        num_years = len(all_stats[0])
        keys = all_stats[0][0].keys()

        avg_stats = []
        min_stats = []
        max_stats = []

        for y in range(num_years):
            yearly_values_list = [sim[y] for sim in all_stats]

            avg_year_dict = {}
            min_year_dict = {}
            max_year_dict = {}

            for k in keys:
                vals = [d[k] for d in yearly_values_list]
                if all(isinstance(v, (int, float, np.float64)) for v in vals):
                    avg_year_dict[k] = float(np.mean(vals))
                    min_year_dict[k] = float(np.min(vals))
                    max_year_dict[k] = float(np.max(vals))
                else:
                    avg_year_dict[k] = yearly_values_list[0][k]
                    min_year_dict[k] = yearly_values_list[0][k]
                    max_year_dict[k] = yearly_values_list[0][k]

            avg_stats.append(avg_year_dict)
            min_stats.append(min_year_dict)
            max_stats.append(max_year_dict)

        avg_population_data = all_population_data[0]

        results_by_immigration[nm_val] = {
            "avg_stats": avg_stats,
            "min_stats": min_stats,
            "max_stats": max_stats,
            "avg_max_count": avg_max_count,
            "avg_max_hist_count": avg_max_hist_count,
            "avg_children_per_female_natives": avg_children_per_female_natives,
            "avg_children_per_female_immigrants": avg_children_per_female_immigrants,
            "avg_children_per_female_mixed": avg_children_per_female_mixed,
            "avg_population_data": avg_population_data,
            "avg_simulation_batch": avg_simulation_batch
        }

    return results_by_immigration
