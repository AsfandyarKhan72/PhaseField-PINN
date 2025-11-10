# PhaseFieldPINN

Physics-Informed Neural Network (PINN) framework for modeling **martensitic phase transformations**.

---

## Overview

This repository contains the codes and data developed for our research on **Physics-Informed Neural Networks (PINNs)** applied to martensitic transformations.  
The work is currently **under review** for publication.

Martensitic transformations are diffusionless phase transitions that occur through lattice distortions without altering the chemical composition, giving rise to unique phenomena such as **superelasticity** and **shape memory effects**.  
Traditional computational methods can efficiently solve **forward problems** (predicting microstructural evolution for given material parameters), but they struggle with **inverse problems**, such as estimating parameters that are difficult to measure experimentally — for example, the **interfacial energy** or **gradient energy coefficients**.

In this study, we develop a **PINN-based framework** that:
- Embeds the **governing physical laws** (phase-field model equations) directly into the network training.
- Enables accurate **forward simulations** of microstructure evolution.
- Allows **inverse estimation** of material parameters (e.g., β) from limited or noisy observed data.

---
