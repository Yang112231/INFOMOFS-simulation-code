### INFOMOFS Smart Charging in Polderwijk

Requirements:
- Python => 3.10.15
- numpy >= 1.25
- pandas >= 1.5
- matplotlib >= 3.7
- scipy >= 1.10 (for truncated normal)

Show help messages:
```bash
 $ python simulatiom.py --help
```
Usage method:
```bash
$ python simulation.py  --strategy <strategy> \
        --scenario <scenario>                 \
        --runs 10                             \
        --horizon 30
```
Command line flags:
 - strategy:     base | price | fcfs | elfs | preempt
 - scenario:     no_solar | p6p7_w | p6p7_s | p1p2p6p7_w | p1p2p6p7_s
 - runs:         independent replications (default 10)
 - horizon:      simulated days per run   (default 30)
 - seed:         RNG seed (optional)

After the simulation two CSV files are generated in output folder:
  1. cables_STRATEGY_SCENARIO.csv: cable statistics (Table 4)
  2. vehicles_STRATEGY_SCENARIO.csv: vehicle statistics (Table 5)

Also several visual plots can be created, see `run_experiment()` function.

The `expand_network.py` file contains the expanded network topology for research questions part. Use the same way as `simulation.py` to run this code, and the corresponding output csv are stored in `expand_output` folder.