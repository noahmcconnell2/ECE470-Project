# ECE470-Project

## Swarm Behaviour Evolution Using Genetic Algorithms

This project explores the evolution of decentralized swarm behavior using a Genetic Algorithm (GA). Agents learn to navigate complex environments by optimizing a set of behavioral heuristics, evolving over generations toward efficient, collision-avoiding, leader-following behavior.

---

## Objectives

- Simulate swarm agents navigating a 2D obstacle-laden grid.
- Evolve swarm behavior using GA based on a weighted combination of features:
  - Cohesion
  - Separation
  - Obstacle Avoidance
  - Path Following
  - Leader Distance
  - Alignment
- Encourage emergent coordination through evolution, not manual rule-coding.

---

## Key Components

### Genetic Algorithm
- Crossover: Simulated Binary Crossover (SBX)
- Mutation: Gaussian mutation (clamped)
- Selection: Tournament
- Diversity Maintenance: K-random genome injection with exponential decay

### Simulation
- Grid size: 50x50 with randomly generated obstacles
- Leader follows an A* path (Chebyshev distance)
- Followers are evaluated on metrics like distance, collisions, and completion status

### Visualization
- Real-time Pygame animation with zoom/restart/recording support
- Optional video export using `imageio`

---

## Project Structure
├── agent/ # Agent class and behavior tracking
├── map/ # Grid, tile, and map generation logic
├── simulation/ # Main simulation, GA, fitness, movement
├── utils/ # Logging, video capture
├── logs/ # Auto-generated output folder
├── configs.py # All tunable settings and constants
├── main.py # Orchestrates training and testing
├── README.md 

---

## Requirements

- Python 3.9+
- numpy
- pygame
- imageio
- deap
- seaborn (optional, for boxplots)
- matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt`
```
---

## Usage

Run the full pipeline (evolution + visualization):
`python main.py`

Outputs (plots, summaries, videos) will be saved to:
`logs/<timestamp>/`


---

## Output

- Fitness convergence plots
- Gene evolution across generations
- Boxplots of simulation metrics
- Correlation plots between gene weights and performance

---

## Notes

- Evolution and testing modes are toggleable via `configs.py`
- TOP_GENOME can be reused across runs for resumed evaluation

---

## Credits

Developed by:
- lexph
- noahmcconnell2
- jacksoneasden
- brandonchiem

Assisted by:
- OpenAI ChatGPT (used in development of plotting and video capture utilities)
- Microsoft Copilot for VSCode (used for documentation and modular refactoring)

---
