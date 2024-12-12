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

# Run the simulation once
(
    stats,
    max_count,
    max_hist_count,
    avg_children_per_female_natives,
    avg_children_per_female_immigrants,
    avg_children_per_female_mixed,
    population_data,
    simulation_batch
) = sim1.run_large_simulation()

# Scale counts
max_count *= simulation_batch
max_hist_count *= simulation_batch

# Prepare data for plotting
years = list(range(len(stats)))
total_population = [stat['total_population'] * simulation_batch for stat in stats]
native_population = [stat['native_population'] * simulation_batch for stat in stats]
immigrant_1_population = [stat['immigrant_1_population'] * simulation_batch for stat in stats]
immigrant_2_population = [stat['immigrant_2_population'] * simulation_batch for stat in stats]
mixed_population = [stat['mixed_population'] * simulation_batch for stat in stats]
immigrant_percentage = [stat['immigrant_percentage'] for stat in stats]

# Collect pyramid data for 100 years
pyramid_data = {}
for year in range(min(100, len(population_data))):  # Ensure we don’t exceed the simulation range
    pyramid_data[year] = {
        "ageGroups": population_data[year]["pyramid_data"]["age_groups"],
        "nativeMaleCounts": [count * simulation_batch for count in population_data[year]["pyramid_data"]["native_male_counts"]],
        "nativeFemaleCounts": [count * simulation_batch for count in population_data[year]["pyramid_data"]["native_female_counts"]],
        "immigrantMaleCounts": [count * simulation_batch for count in population_data[year]["pyramid_data"]["immigrant_male_counts"]],
        "immigrantFemaleCounts": [count * simulation_batch for count in population_data[year]["pyramid_data"]["immigrant_female_counts"]]
    }

# Determine the last year index from the years list
last_year = years[-1]

# Access the gene values from the last year's data
gene_values = population_data[last_year]["gene_values"]  # if population_data is keyed by year

bins = np.arange(0, 101, 10)
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

geneHistogramData = {}
for year in range(100):
    gene_values = population_data[year]["gene_values"]
    # Compute histogram just like before, but per year
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
        "bins": list(bins),
        "nativeGeneCounts": native_counts,
        "immigrant1GeneCounts": imm_1_counts,
        "immigrant2GeneCounts": imm_2_counts
    }

# Compile final data
data = {
    "years": years,
    "totalPopulation": total_population,
    "nativePopulation": native_population,
    "immigrant1Population": immigrant_1_population,
    "immigrant2Population": immigrant_2_population,
    "mixedPopulation": mixed_population,
    "immigrantPercentage": immigrant_percentage,
    "pyramidData": pyramid_data,  # Dictionary of 100 years of pyramid data
    "geneHistogramData": geneHistogramData,
    "avgChildrenPerFemaleNatives": avg_children_per_female_natives,
    "avgChildrenPerFemaleImmigrants": avg_children_per_female_immigrants,
    "avgChildrenPerFemaleMixed": avg_children_per_female_mixed
}

# Save to JSON
output_file = "data.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=4, default=convert_numpy)

print(f"Simulation data for 100 years exported to {output_file}")
