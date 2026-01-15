
# Proof Number applied to MCTS in combination with MAST

*Technical University of Berlin*

This repository contains the source code and experimental data for the Bachelor's thesis, exploring enhancements to the **Proof-Number Monte Carlo Tree Search (PN-MCTS)** algorithm by David Celik supervised by Dr.- Ing. Stefan Fricke. This project is built upon the original Java implementation of PN-MCTS by Kowalski et al. (2024).

## Abstract

Modern game-tree search often involves a trade-off between the statistical foresight of Monte Carlo Tree Search (MCTS) and the exact solvability provided by Proof Number Search (PNS). While MCTS excels at general strategy through randomized exploration, it can be inefficient in tactical situations where PNS is designed to find concrete proofs. Proof-Number Monte Carlo Tree Search (PN-MCTS) addresses this by embedding logical proof numbers directly into the probabilistic framework, creating a hybrid architecture capable of navigating both strategic and tactical search spaces.

Despite this, the standard PN-MCTS framework relies on random simulations that do not fully leverage search data. This thesis aims to enhance the algorithm's performance by integrating:
* **MAST** (Move Average Sampling Technique)
* **NST** (N-gram Selection Technique)
* **RAVE** (Rapid Action Value Estimation)
* **GRAVE** (Generalized RAVE)

The research investigates whether biasing simulations with historical knowledge can improve decision quality and if these statistical heuristics can serve as a viable alternative to complex structural logic for handling game outcomes.

Using a testbed of four distinct domains—**Awari, Lines of Action, Knightthrough, and MiniShogi**—this work evaluates the impact of these enhancements on search efficiency. The results demonstrate a clear divergence between heuristics applied to the tree policy versus those applied to the playout policy. Explicitly biasing node selection via RAVE and GRAVE proved significantly more effective than implicitly influencing the stochastic playout phase with MAST or NST.

Furthermore, the empirical analysis reveals that while statistical heuristics optimize search direction, they cannot replace the structural **L2 layer** of proof numbers required to logically differentiate draws from losses. Consequently, this thesis concludes that strong configuration for PN-MCTS is a symbiotic integration: utilizing the L2 layer for the structural resolution of game theoretic values, while employing RAVE-based mechanisms to guide the search through complex tactical branches.

## Reference & Acknowledgements

This work is based on the research and original code by **Jakub Kowalski, Elliot Doe, Mark H. M. Winands, Daniel Górski, and Dennis J. N. J. Soemers**.

**Original Article:**
[**Proof Number Based Monte-Carlo Tree Search**](https://doi.org/10.1109/TG.2024.3403750) (*IEEE Transactions on Games*, 2024). An extended version of the paper is available on [ArXiv](https://arxiv.org/abs/2303.09449).

If you use the base PN-MCTS algorithm found in this repository, please cite the original paper:

```bibtex
@article{Kowalski2024ProofNumber,
  author = {Kowalski, J. and Doe, E. and Winands, M. H. M. and G\'{o}rski, D. and Soemers, D. J. N. J.},
  title = {{Proof Number Based Monte-Carlo Tree Search}},
  journal = {IEEE Transactions on Games},
  volume = {},
  number = {},
  pages = {1--10},
  year = {2024},
}------------------------------------