import json
import numpy as np
import sim1  # Your simulation module

def convert_numpy(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    return o

# Run multiple simulations to get average, min, and max stats, and fertility arrays
(
    stats,
    min_stats,
    max_stats,
    max_count,
    max_hist_count,
    avg_children_per_female_natives,
    avg_children_per_female_immigrants,
    avg_children_per_female_mixed,
    population_data,
    simulation_batch
) = sim1.monte_carlo_simulations(10)  # For example, 10 simulations

# Extract years
years = list(range(len(stats)))

# Create arrays for total population from average stats
total_population = [stat['total_population'] * simulation_batch for stat in stats]

# Create min/max arrays for total population
min_total_population = [stat['total_population'] * simulation_batch for stat in min_stats]
max_total_population = [stat['total_population'] * simulation_batch for stat in max_stats]

# Other population groups (just average, no min/max needed)
native_population = [stat['native_population'] * simulation_batch for stat in stats]
immigrant_1_population = [stat['immigrant_1_population'] * simulation_batch for stat in stats]
immigrant_2_population = [stat['immigrant_2_population'] * simulation_batch for stat in stats]
mixed_population = [stat['mixed_population'] * simulation_batch for stat in stats]
immigrant_percentage = [stat['immigrant_percentage'] for stat in stats]

# Prepare pyramid data (average scenario)
pyramid_data = {}
for year in range(min(100, len(population_data))):
    pyramid = population_data[year]["pyramid_data"]
    pyramid_data[year] = {
        "ageGroups": pyramid["age_groups"],
        "nativeMaleCounts": [count * simulation_batch for count in pyramid["native_male_counts"]],
        "nativeFemaleCounts": [count * simulation_batch for count in pyramid["native_female_counts"]],
        "immigrantMaleCounts": [count * simulation_batch for count in pyramid["immigrant_male_counts"]],
        "immigrantFemaleCounts": [count * simulation_batch for count in pyramid["immigrant_female_counts"]]
    }

# Prepare gene histogram data (average scenario)
bins = np.arange(0, 101, 10)
geneHistogramData = {}
for year in range(100):
    if year not in population_data:
        break
    gene_values = population_data[year]["gene_values"]
    native_counts = [0]*11
    imm_1_counts = [0]*11
    imm_2_counts = [0]*11
    for gv in gene_values:
        n_bin = int(gv.get("native", 0) / 10)
        i1_bin = int(gv.get("immigrant_1", 0) / 10)
        i2_bin = int(gv.get("immigrant_2", 0) / 10)

        native_counts[n_bin] += simulation_batch
        imm_1_counts[i1_bin] += simulation_batch
        imm_2_counts[i2_bin] += simulation_batch

    geneHistogramData[year] = {
        "bins": bins.tolist(),
        "nativeGeneCounts": native_counts,
        "immigrant1GeneCounts": imm_1_counts,
        "immigrant2GeneCounts": imm_2_counts
    }

# Compile final data
data = {
    "years": years,
    # Population arrays with min/max for total population only
    "totalPopulation": total_population,
    "minTotalPopulation": min_total_population,
    "maxTotalPopulation": max_total_population,

    # Just average arrays for other population groups
    "nativePopulation": native_population,
    "immigrant1Population": immigrant_1_population,
    "immigrant2Population": immigrant_2_population,
    "mixedPopulation": mixed_population,
    "immigrantPercentage": immigrant_percentage,

    # Pyramid and gene histograms (no min/max needed here)
    "pyramidData": pyramid_data,
    "geneHistogramData": geneHistogramData,

    # Ordinary fertility curves (averages only, as requested)
    "avgChildrenPerFemaleNatives": avg_children_per_female_natives,
    "avgChildrenPerFemaleImmigrants": avg_children_per_female_immigrants,
    "avgChildrenPerFemaleMixed": avg_children_per_female_mixed
}

# Save to JSON
output_file = "data.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=4, default=convert_numpy)

print(f"Simulation data with min/max for total population and average fertility exported to {output_file}")
