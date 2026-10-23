# Exact OT-CFM: baseline architecture

Use exact minibatch source–target assignment while retaining the baseline architecture. This is distinct from the exploratory multiscale OT run.

## Included evidence

Full group-disjoint test metrics for 1,053 images and evaluation provenance. The training notebook is included; checkpoints and full predictions are external.

## Run

Open [ot-cfm.ipynb](ot-cfm.ipynb) in Kaggle. Attach the [prepared MIST-HER2 bundle](https://www.kaggle.com/datasets/vasiliygorelov/mist-her2), enable the accelerator required by the notebook and run cells in order. Review the first configuration cell before launching training or test generation.

The `ihc_flow` source package is loaded from the dataset bundle, not installed from this repository. Keep the bundle version and split fixed. Smoke modes only check code and do not produce meaningful quality scores.

## Hosted runs

- [Kaggle run 1](https://www.kaggle.com/code/marritau/ot-cfm)
- [Kaggle run 2](https://www.kaggle.com/code/marritau/ot-cfm-evaluation)

## Results

[Saved logs, configuration and evaluation tables](results/) are extracted from the original archives or notebook output. Raw archive checksums are recorded in `docs/local-archives.json` at the repository root.

## Interpretation

Use the manifest to identify train/validation/test. Original `train/` and `test/` prefixes in image IDs are release paths. Do not compare small validation galleries, selected test subsets and full-test results as if they used the same evaluation population.

## Notebook output files

[Figures and original console output](results/notebook-outputs/README.md) are stored separately so the source notebook stays easy to browse. The full executed copy is retained locally as `*.executed.ipynb` and excluded from Git. Code cell source is unchanged.
