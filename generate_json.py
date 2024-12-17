import json
import numpy as np
import sim1  # Your simulation module that returns the dictionary

def convert_numpy(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    return o

# Run simulations and get a dictionary keyed by net_migration scenario
results_by_immigration = sim1.monte_carlo_simulations(10)

# Loop over all scenarios and process them
all_data = {}

for scenario_key in results_by_immigration.keys():
    scenario_data = results_by_immigration[scenario_key]

    # Extract data from the current scenario
    stats = scenario_data["avg_stats"]
    min_stats = scenario_data["min_stats"]
    max_stats = scenario_data["max_stats"]
    max_count = scenario_data["avg_max_count"]
    max_hist_count = scenario_data["avg_max_hist_count"]
    avg_children_per_female_natives = scenario_data["avg_children_per_female_natives"]
    avg_children_per_female_immigrants = scenario_data["avg_children_per_female_immigrants"]
    avg_children_per_female_mixed = scenario_data["avg_children_per_female_mixed"]
    population_data = scenario_data["avg_population_data"]
    simulation_batch = scenario_data["avg_simulation_batch"]

    # Define the years based on the length of stats
    years = list(range(len(stats)))

    # Create population arrays using the extracted stats and simulation_batch
    total_population = [stat['total_population'] * simulation_batch for stat in stats]
    min_total_population = [stat['total_population'] * simulation_batch for stat in min_stats]
    max_total_population = [stat['total_population'] * simulation_batch for stat in max_stats]

    native_population = [stat['native_population'] * simulation_batch for stat in stats]
    immigrant_1_population = [stat['immigrant_1_population'] * simulation_batch for stat in stats]
    immigrant_2_population = [stat['immigrant_2_population'] * simulation_batch for stat in stats]
    mixed_population = [stat['mixed_population'] * simulation_batch for stat in stats]
    immigrant_percentage = [stat['immigrant_percentage'] for stat in stats]

    # Prepare pyramid data (using integer keys)
    pyramid_data_converted = {}
    for year in range(min(100, len(population_data))):
        pyramid = population_data[year]["pyramid_data"]
        pyramid_data_converted[year] = {
            "ageGroups": pyramid[0],
            "nativeMaleCounts": [count * simulation_batch for count in pyramid[1]],
            "immigrantMaleCounts": [count * simulation_batch for count in pyramid[2]],
            "nativeFemaleCounts": [count * simulation_batch for count in pyramid[3]],
            "immigrantFemaleCounts": [count * simulation_batch for count in pyramid[4]]
        }

    # Prepare gene histogram data
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

    # Compile data for the current scenario
    data = {
        "years": years,
        "totalPopulation": total_population,
        "minTotalPopulation": min_total_population,
        "maxTotalPopulation": max_total_population,
        "nativePopulation": native_population,
        "immigrant1Population": immigrant_1_population,
        "immigrant2Population": immigrant_2_population,
        "mixedPopulation": mixed_population,
        "immigrantPercentage": immigrant_percentage,
        "pyramidData": pyramid_data_converted,
        "geneHistogramData": geneHistogramData,
        "avgChildrenPerFemaleNatives": avg_children_per_female_natives,
        "avgChildrenPerFemaleImmigrants": avg_children_per_female_immigrants,
        "avgChildrenPerFemaleMixed": avg_children_per_female_mixed
    }

    # Add the processed scenario to the results
    all_data[scenario_key] = data

# Save all scenarios to JSON
output_file = "all_scenarios_data.json"
with open(output_file, "w") as f:
    json.dump(all_data, f, indent=4, default=convert_numpy)

print(f"Exported all scenario data to {output_file}")
