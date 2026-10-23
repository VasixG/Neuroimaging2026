# HER2match experiments

Open [`her2match_experiments.ipynb`](her2match_experiments.ipynb) for the five experiments from the final presentation, recorded results, learning curves, and fixed image comparisons:

1. Noise CFM
2. Direct CFM
3. Direct CFM + LPIPS
4. Direct CFM + stain-aware sampling
5. Direct CFM + staining-sensitive loss

No optimal-transport reassignment or bridge experiments are included. Training and evaluation use aligned HER2match only. Evaluation uses the same 252 pairs, 63 per slide-level HER2 score.

The notebook defaults to **review mode**: it displays completed-run outputs without training. Source code and the required `ihc_flow` components are included in `her2match/`; no external source bundle is required.

### Rerun

Run commands from the `HER2Match/` directory.

Install a suitable CUDA build of PyTorch and the dependencies in `requirements-her2match.txt`. Set `HER2_DATA_ROOT` to the dataset directory containing `tiles/HE/{train,val,test}` and `tiles/IHC/{train,val,test}`. Set `CUDA_VISIBLE_DEVICES` to your allocated GPU before starting Python, then enable `RUN_TRAINING` in the notebook. The five runs execute sequentially. Keep the defaults to match the submitted protocol.

Alternatively:

```bash
python her2match/baseline.py
python her2match/fixes.py scan
python her2match/fixes.py run
```

New artifacts go to `outputs/` (gitignored). Override `HER2_BASELINE_OUT` and `HER2_FIX_OUT` if needed. Training uses 20,000 optimizer updates per run and saves EMA-best and resumable-latest checkpoints. Data, checkpoint binaries, pretrained weights, and complete prediction collections are not stored in Git. Small presentation image panels and recorded metrics/histories are included under `results/her2match/`.

Portable code has new configuration identities; the recorded server checkpoint hashes are provenance, not compatible resume configurations. Do not mix archived checkpoints with new runs. The notebook is a consolidated review/execution workflow, not a claim that the original server jobs ran inside its cells.

### Packaging validation

The submission was checked by executing the notebook in review mode, validating the manifests and saved metrics, and checking the portable model/data/stain helpers. Full training was not repeated for packaging.
