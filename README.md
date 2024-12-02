# Agent-Based Population Simulation

This project simulates population dynamics using an agent-based model, visualized using Dash and Plotly. The simulation models aging, reproduction, death, and migration of individuals over time, providing insights into how populations evolve under various demographic factors.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Simulation Details](#simulation-details)
- [Visualization](#visualization)
- [License](#license)

## Introduction

This agent-based simulation models a population over a specified number of years, considering realistic demographic factors such as:

- Age distributions based on real-world data.
- Age-dependent fertility and mortality rates.
- Immigration with specific age and fertility characteristics.
- Genetic inheritance modeling (e.g., immigrant status as a gene).

The simulation results are visualized using an interactive Dash application, allowing users to explore various population statistics and trends dynamically.

## Features

- **Agent-Based Modeling**: Each individual is an agent with attributes like age, sex, fertility rate, and genes.
- **Dynamic Fertility and Mortality Rates**: Fertility and mortality probabilities change based on age, number of children, and immigrant status.
- **Migration Modeling**: Incorporates net migration, adding immigrants with different characteristics to the population annually.
- **Interactive Dash Application**: Visualize population trends, age-sex pyramids, fertility rates, and genetic distributions interactively.

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Dependencies

Install the required Python packages using pip:

```bash
pip install numpy dash plotly tqdm
