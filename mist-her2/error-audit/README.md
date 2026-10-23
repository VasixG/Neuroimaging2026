# Dataset and prediction audit

Inspect alignment proxies, DAB strata, high-stain failures, false staining, blur and morphology mismatch.

## Included evidence

The multi-model audit compares baseline-256 and VP-CFM on 1,053 IDs at audit resolution 256. These are diagnostic metrics, not headline full-resolution scores.

## Run

Open [error-audit.ipynb](error-audit.ipynb) in Kaggle. Attach the [prepared MIST-HER2 bundle](https://www.kaggle.com/datasets/vasiliygorelov/mist-her2), enable the accelerator required by the notebook and run cells in order. Review the first configuration cell before launching training or test generation.

CPU is sufficient. Also attach complete baseline-256 and VP-CFM prediction sets; the included reports alone cannot regenerate per-image measurements. Use the discovery overrides if automatic input discovery is ambiguous.

The `ihc_flow` source package is loaded from the dataset bundle, not installed from this repository. Keep the bundle version and split fixed. Smoke modes only check code and do not produce meaningful quality scores.

## Hosted runs

- [Kaggle run 1](https://www.kaggle.com/code/marritau/error-audit)

## Results

[Saved logs, configuration and evaluation tables](results/) are extracted from the original archives or notebook output. Raw archive checksums are recorded in `docs/local-archives.json` at the repository root.

## Interpretation

Use the manifest to identify train/validation/test. Original `train/` and `test/` prefixes in image IDs are release paths. Do not compare small validation galleries, selected test subsets and full-test results as if they used the same evaluation population.

## Notebook output files

[Figures and original console output](results/notebook-outputs/README.md) are stored separately so the source notebook stays easy to browse. The full executed copy is retained locally as `*.executed.ipynb` and excluded from Git. Code cell source is unchanged.
