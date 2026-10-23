# Flow Matching for Virtual Staining

NeuroML 2026 final project · Staining Flow Swimmers

## Summary

We investigated conditional Flow Matching for translating H&E into HER2 immunohistochemistry. We compared probability paths, crop sizes, transport coupling, pathology-feature guidance and training-pair sampling. The main failures were weak dependence on H&E, lost morphology and missed or misplaced brown staining. Direct flow showed stronger input dependence in an inspected example but did not reliably reproduce realistic HER2 staining. Saved metrics and diagnostic galleries support a distinction between paired similarity, image realism and staining accuracy.

We focus this report on our MIST-HER2 experiments and the results recorded in this repository.

## Task and research questions

The input is a routine H&E image and the target is a corresponding real HER2 image. We aim to preserve tissue structure while reproducing the amount and location of staining. We ask whether changing the transport path or source, strengthening conditioning, or changing pair sampling improves these properties. Computational cost is a secondary question, explored through limited solver comparisons.

## Data, preprocessing and exploration

The prepared [MIST-HER2 Kaggle bundle](https://www.kaggle.com/datasets/vasiliygorelov/mist-her2) provides paired RGB images, runtime code and manifests. The group-disjoint split contains 3,805 train, 784 validation and 1,053 test pairs. Groups are inferred from filenames and are not verified patient identities. Release-directory prefixes in IDs are independent of the current manifest split.

The source images are 1024×1024. Baseline and several variants use native 512-pixel training crops; the crop-size study uses 256-pixel crops. Direct-Phikon uses crop512 followed by resize256. Tensor scaling and crop handling are implemented in the attached bundle; the notebook configuration records the selected recipe. No single preprocessing protocol should be assumed for every experiment.

Data exploration includes split counts, group overlap, stain distributions and alignment proxies. The dataset audit covers all 1,053 test IDs. DAB strata describe color signal, not clinical HER2 grades. Alignment measures describe cross-stain morphology agreement, not known deformation fields. Diagnostic strata must not replace the primary unfiltered test evaluation.

## Models and optimization

The baseline U-Net receives the current state, fixed H&E condition and time, then predicts velocity. Linear CFM uses `x_t = (1-t)z + ty` and target velocity `y-z`. Generation integrates the learned velocity from noise while retaining H&E conditioning. Real IHC is used as a training target and is not available at inference.

VP-CFM replaces the linear path with a trigonometric path. Exact OT-CFM uses a minimum-cost source–target assignment in a minibatch with the baseline architecture. A separate exploratory guided OT run also changes the H&E encoder; its effects cannot be attributed to OT alone. Phikon guidance combines frozen pathology features with local features and auxiliary content, structure, semantic and DAB objectives. Direct-Phikon starts from H&E plus small noise and uses additional endpoint supervision on an actual rollout.

Runs use AdamW, warmup/cosine schedules and EMA where specified in their saved recipes. Baseline configuration records learning rate 1e-4, warmup 500 and EMA 0.999. Direct-Phikon's stable recipe records learning rate 3e-5 and EMA 0.995. Checkpoint selection differs: several runs use validation SSIM, Phikon guidance uses a structure/DAB proxy, and direct-Phikon uses LPIPS plus DAB error. These differences are part of the experimental context, not an isolated loss ablation.

## Evaluation and results

[Experiment tracking](experiment-tracking.md) links every summary below to its source files. SSIM measures paired agreement, while FID and KID describe distribution differences. DAB statistics and visual inspection investigate staining errors. None of these alone demonstrates correct membrane localization.

| Method | Evaluation | SSIM ↑ | FID ↓ | KID ×1000 ↓ |
|---|---|---:|---:|---:|
| Exact OT-CFM | Full test, 1,053 images | 0.1682 | 186.58 | 191.55 |
| VP-CFM | Full test, 1,053 images | 0.1447 | 367.41 | 469.09 |
| DAB-balanced | Selected test subset, 350 images | 0.1694 | 365.75 | 454.73 |
| DAB + alignment-balanced | Same selected subset, 350 images | 0.1757 | 327.90 | 382.83 |

The two full-test archives record Heun16, 32 NFE and seed 42. They are not automatically comparable with published official-split scores. The selected 350-image DAB subset must not be ranked against the full-test population. Within that selected subset, alignment-aware sampling has better point estimates of SSIM, FID and KID, while mean DAB-fraction error is nearly unchanged (0.13151 versus 0.13142). This is limited evidence, not proof of a general biological improvement.

Small Phikon and direct-flow metric files cover 12 and 8 validation crops. They are exploratory previews. We present the crop256 comparison as exported figures and restrict the table above to runs with machine-readable evaluation records.

## Interpretation and error analysis

Condition swaps keep initial noise and model settings fixed while changing H&E. Similar outputs suggest weak use of the condition. A different output only establishes sensitivity; its geometry must also follow the new input. Seed swaps inspect variation at fixed H&E and do not quantify calibrated biological uncertainty.

The [multi-model audit](../mist-her2/error-audit/) compares baseline256 and VP-CFM predictions over the same 1,053 IDs at 256-pixel audit resolution. Failure galleries distinguish high-DAB underestimation, extra brown staining, blur, morphology mismatch and poorly aligned pairs. The main visual finding is that generating IHC-like texture or more brown color is insufficient: membrane patterns can still be missing or misplaced.

Phikon guidance sometimes increased SSIM while DAB and structure proxies worsened. Direct transport made input dependence more visible in an inspected example, but generated images often appeared mask-like. Poor pair alignment is a plausible contributor; it is not the only explanation, and cross-dataset comparisons do not isolate its causal effect.

## Progress against the proposal

| Task | Evidence and status |
|---|---|
| a. Review FM and inspect MIST | Implemented; method notebooks, paired examples and dataset audit |
| b. Implement and train conditional FM | Implemented; saved training histories and configurations. Solver tuning is limited |
| c. Benchmark and compare with literature | Partial: full-test OT/VP metrics are available; protocol differences limit literature comparison and baseline assets are incomplete |
| d*. Solver/time-quality study | Partial: limited validation comparisons, not a broad controlled benchmark |
| e*. Marker-specific failure analysis | Partial: visual audits and DAB proxies, without validated cell/membrane annotations |

No numerical completion percentage is assigned without an agreed task weighting. The archived duplicate OT notebook is excluded from the count of independent experiments.

## Limitations and next steps

Several development stages changed architecture, source, loss and sampling together. Run lengths and checkpoint criteria also differ. Small validation subsets, image misalignment and imperfect stain proxies limit interpretation. Some pretrained components and runtime source are loaded from external Kaggle Inputs, while checkpoints and full prediction sets are managed separately from this source repository.

Next, isolate one change per experiment under a shared validation and sampling protocol. Evaluate stain-positive and stain-negative cases separately, report uncertainty across groups and seeds, and measure quality against NFE and inference time. A controlled alignment study and annotated cell-level evaluation are needed before claiming reliable HER2 localization.

## Reproducibility and contributions

The [reproduction guide](reproduction.md) documents required Inputs, environment setup, checkpoint behavior and external run dependencies. The [tracking index](experiment-tracking.md) provides original configurations, JSONL histories, metrics, per-image tables and figures. We store source notebooks alongside figures and text outputs in `results/notebook-outputs/`. Full executed copies remain local and are excluded from Git.

Vasilii Gorelov worked on Flow Matching implementation and modeling; Andrey Galichin on data preparation and evaluation; Margarita Urazmetova on experiments, sampling and failure analysis. All members contributed to interpretation, the report and presentation.

## References

- [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- [Liu et al., Flow Straight and Fast](https://arxiv.org/abs/2209.03003).
- [Tong et al., Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482).
- [Li et al., Adaptive Supervised PatchNCE / MIST](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE).
- [Phikon model card](https://huggingface.co/owkin/phikon).

Third-party implementations retain their provenance records in the corresponding notebook. Follow dataset, source-code and pretrained-weight licenses when reusing them.
