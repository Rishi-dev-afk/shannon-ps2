# Knowledge PET3D data and RF provenance

## Source and attribution

Use the trajectory and published conflict CSVs from [Knowledge PET3D on Mendeley Data](https://data.mendeley.com/datasets/2wtbb9jnpx/1), version 1. The official analysis code and data layout notes are in the [Knowledge PET3D repository](https://github.com/4rnd25/knowledge_PET3D).

> Pettirsch, A. and Garcia-Hernandez, A. (2025). “Knowledge PET3D: An interpretable framework for 3D near-miss detection in thermal traffic video.” Expert Systems with Applications, 130821. https://doi.org/10.1016/j.eswa.2025.130821  
> Dataset DOI: https://doi.org/10.17632/2wtbb9jnpx.1

The dataset is published under CC BY 4.0 according to the accepted project plan. Retain attribution and the license link when redistributing derived materials. This repository stores neither the 2.63 GB source archive nor the raw videos.

## Local layout

Extract the archive outside Git, retaining the two location directories, for example:

    data/Knowledge_PET3D_3D_Traffic_Trajectories_test/
      Location_A_Trajectories_and_Conflicts/
        ... trajectory CSV files ...
        ... *Conflicts.csv files ...
      Location_B_Trajectories_and_Conflicts/
        ... trajectory CSV files ...
        ... *Conflicts.csv files ...

The preparation command discovers CSVs by their fields. Trajectory CSVs need idVehicle, FrameTimeStamp (or TimeStamp), posXFit, and posYFit. Conflict CSVs need idVehicle1, idVehicle2, and a timestamp. Session identity is formed from idMeasurementSeries, idSensor, and idRecord; keep these values consistent between the trajectory and conflict files.

Prepare pair windows:

    python -m vision_zero.cli prepare-risk-data data/Knowledge_PET3D_3D_Traffic_Trajectories_test outputs/training/knowledge-pet3d-pair-windows.csv

Train and evaluate the Random Forest:

    python -m vision_zero.cli train-risk-model outputs/training/knowledge-pet3d-pair-windows.csv --model-out outputs/models/knowledge-pet3d-rf.joblib

The training command writes a compact compressed model artifact, an adjacent feature schema, and held-out evaluation JSON. The two location folds are Location_A→Location_B and Location_B→Location_A. Threshold tuning uses recording groups from the training location only.

## Target and limitations

The target is whether the published Knowledge PET3D method has a conflict record for the vehicle pair in the following two seconds, using one second of prior trajectory history. PET, LOF, conflict scores, direct source flags, vehicle IDs, recording IDs, and location are excluded from model features.

A label of zero means only “not flagged by the source method.” It is not a manually verified safe example. The model reproduces source-method flags and must not be described as predicting crashes or safety outcomes. Results from only two locations are location-held-out checks, not broad geographic generalization.

The prepared pair-window CSV can be large and includes source-derived labels. Keep it local unless the dataset attribution and derived-data redistribution terms have been reviewed. Do not commit the source archive, full trajectory exports, source videos, or detector weights.
