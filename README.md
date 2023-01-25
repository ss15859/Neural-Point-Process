# Forecasting the 2016-2017 Central Apennines Earthquake Sequence with a Neural Point Process

Accompanying source code for [manuscript](https://arxiv.org/abs/2301.09948).

## Setup

Create and activate python environment:

```bash
bash setup.sh
conda activate py-37-keras-21
```

## To generate figures from the manuscript
```bash
python plot_AVN_catalog.py
```
![Alt text](Dataset_with_completeness.png?raw=true "Overview")

Results on the simulated datasets:
```text
python plot_simulated_results.py complete
python plot_simulated_results.py incomplete
```
Results on the Central Apennines catalog:
```bash
python plot_AVN_results.py Visso
python plot_AVN_results.py Norcia
python plot_AVN_results.py Campotosto
```
Cumulative Information Gain (CIG) plot:
```bash
python plot_CIG_over_time.py Norcia
```
## To retrain and evaluate models

Move results to temporary files:
```bash
bash move_results.sh
```
Run experiments: (Warning! the smaller the magnitude threshold (Mcut) the longer the code will take to run)
```bash
python simulated_experiment.py [complete/incomplete] [Mcut]
python AVN_experiment.py [Visso/Norcia/Campotosto] [Mcut]
```

## To generate figures from the supporting information

Uncomment the parameter plot section and run:

```bash
python plot_AVN_results.py Visso
python plot_AVN_results.py Norcia
python plot_AVN_results.py Campotosto
```
Then run:
```text
python plot_simulated_CIG_over_time.py complete
python plot_simulated_CIG_over_time.py incomplete
```

