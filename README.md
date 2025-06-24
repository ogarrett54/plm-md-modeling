# Training a PLM to predict biophysical metrics from MD simulations

## Objective

The purpose of this project is to train a model to predict biophysical metrics computed from molecular dynamics simulations. This will enable encoding of biphysical information from sequence alone, which may then be leveraged for fine-tuning on phenotype prediction tasks. The ultimate goal is to leverage this synthetic data source to pretrain a model on Pik1 and AVR-Pik interactions, such that fine-tuning the model on wet lab immunogenicity data requires relatively fewer data points.

## Resources

This work takes inspiration from two key papers:

- [Learning Biophysical Dynamics with Protein Language Models](https://www.biorxiv.org/content/10.1101/2024.10.11.617911v3)
- [Biophysics-based protein language models for protein engineering](https://www.biorxiv.org/content/10.1101/2024.03.15.585128v3)

## Approach

### Pipeline Overview

1. Input PDB file of the wild-type complex, specify which chain to mutate.
2. Extract sequences and fix PDB file.
3. Generate mutant sequences.
4. Use PyRosetta to generate mutant structures for each sequence.
5. Simulate each mutant structure using OpenMM for 100 ns.
6. Calculate pairwise and residue-specific metrics using MDTraj and MDAnalysis.
   - Pairwise
     - residue movement correlation
     - hydrogen bonds
     - salt bridges
     - Van der Waals
     - pi-interaction
     - T-stacking
     - hydrophobic contacts
   - Residue-level
     - residue fluctuation
     - SASA
     - secondary structure
     - dihedral angles
7. Train ESMDance architecture to predict each of these metrics from input sequences.
8. Fine-tune pretrained model with a regression head to predict binding and/or immunogenecity.
