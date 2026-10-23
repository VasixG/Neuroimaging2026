# Reproduction guide

## Inspect saved results without training

Open [experiment tracking](experiment-tracking.md). It links to JSON metrics, configurations, JSONL histories, per-image CSVs and figures. These files are extracted from saved outputs and can be inspected on GitHub. Figures and console output are stored separately in each notebook folder under `results/notebook-outputs/`. The source notebooks have clean outputs; full executed copies are retained locally as `*.executed.ipynb` and excluded from Git.

## Training environment

1. Clone the repository and open the chosen notebook in Kaggle.
2. Attach [vasiliygorelov/mist-her2](https://www.kaggle.com/datasets/vasiliygorelov/mist-her2). Check that the bundle includes `kaggle_bundle.json`, `code/ihc_flow/`, `data/HER2/` and `manifests/group-disjoint/pairs.jsonl`.
3. Enable the notebook's accelerator. Baseline/preview runs use one GPU; exact-OT full-test generation and DAB comparison include two-GPU workers. CPU is enough for error auditing when complete predictions are supplied.
4. Read the first configuration cell. Set dataset paths and any pretrained-weight paths for your own Inputs. Some notebooks find the bundle automatically; others use the Kaggle Input paths from our runs.
5. Run cells in order. Setup cells provide notebook-specific dependencies. Do not replace the Kaggle Torch/CUDA installation with an unrelated environment. The bundle contains runtime source not present in this repository.
6. Save Kaggle Outputs. Keep configs, split fingerprints, checkpoints, histories and evaluation records together.

We identify evaluation inputs through the manifest and checkpoint hashes in saved run metadata. Use these hashes when matching a rerun to the reported results.

## Checkpoints and weights

`RESUME_FROM` restores the state of a matching run. `INITIALIZE_FROM`, where supported, loads starting weights and resets optimizer state. We set it to `None` in the direct-Phikon run. Our Phikon-guidance notebook uses a Kaggle Input path for `PHIKON_PATH`: attach the correct model and change the path or use its supported discovery/download option. Preserve the model's identity and license.

The DAB notebook is a resume workflow and needs two previous `latest.pt` files. Its training ZIP explicitly excludes weights. Recreating the original run requires those external Inputs; the logs alone cannot resume training.

## Evaluation

Choose checkpoints and solver settings on validation. Run test generation only after that choice. Real and generated images must use the same preprocessing and resolution. Record split, count, seed, solver, NFE, crop/resize, metric protocol and checkpoint hash.

The archived exact-OT and VP results evaluate 1,053 full-resolution group-disjoint test pairs with Heun16 (32 NFE), seed 42. DAB-aware sampling was evaluated on a selected 350-image test subset. Audit metrics are calculated at 256 pixels and are separate from full-resolution headline scores. Twelve- and eight-crop galleries are validation previews.

## Run assets

We keep source notebooks, evaluation tables, training histories and selected figures in this repository. Training and inference also use the prepared MIST bundle, pretrained feature weights and run-specific checkpoints. Attach the corresponding Kaggle Inputs/Outputs when resuming a run or regenerating full predictions. The DAB comparison resumes two existing checkpoints; its log archive contains configuration and history rather than weights.
