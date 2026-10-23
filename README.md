# Flow Matching for Virtual Staining

**NeuroML 2026 final project · Staining Flow Swimmers**

We study H&E-to-HER2 image translation with Flow Matching. The experiments test whether changes to the transport path, image conditioning and training-pair sampling improve tissue structure and staining. This repository documents our **MIST-HER2 experiments**.

## Start here

- **[Experiment tracking: runs, configs, logs, metrics and artifacts](docs/experiment-tracking.md)**
- **[Written report](docs/report.md)**
- **[Presentation](docs/NEUROML-2.pdf)**
- [Reproduction guide](docs/reproduction.md)
- [Results with source paths](docs/results-summary.json)

## Experiments

| Folder | Question | Evidence |
|---|---|---|
| [baseline](mist-her2/baseline/) | Does noise-conditioned CFM use H&E effectively? | Notebook logs and example images |
| [crop-256](mist-her2/crop-256/) | Does a smaller crop improve local agreement? | Exported metric and DAB figures |
| [vp-cfm](mist-her2/vp-cfm/) | Does a variance-preserving path help? | Training history and full-test metrics |
| [exact-ot-cfm](mist-her2/exact-ot-cfm/) | Does exact minibatch OT help with the baseline architecture? | Notebook and full-test metrics |
| [ot-cfm-preview](mist-her2/ot-cfm-preview/) | Does OT plus multiscale H&E guidance help? | 12 validation examples/metrics |
| [phikon-guidance](mist-her2/phikon-guidance/) | Can pathology features and structure/stain losses help? | 12 validation examples/metrics |
| [direct-phikon](mist-her2/direct-phikon/) | What changes when generation starts from H&E? | 8 validation examples/metrics and limited solver check |
| [dab-sampling](mist-her2/dab-sampling/) | Does balancing stain and alignment proxies help? | Two training histories; selected 350-image test evaluation |
| [error-audit](mist-her2/error-audit/) | Which failures are hidden by average metrics? | Dataset audit, stratified metrics and failure galleries |

The former `base_direct_OT` folder is retained in [archive/ot-cfm-duplicate](mist-her2/archive/ot-cfm-duplicate/). Its code and exported examples duplicate the exploratory noise-source OT run; it is not another direct-flow result. Original-to-current paths are recorded in [file-migration.json](docs/file-migration.json).

## Data and setup

Use the prepared [MIST-HER2 Kaggle dataset](https://www.kaggle.com/datasets/vasiliygorelov/mist-her2) (`vasiliygorelov/mist-her2`). In Kaggle, attach the dataset as Input and import the experiment notebook. It contains paired images, split manifests and the `ihc_flow` source package used by the notebooks. 

The default group-disjoint split contains 3,805 training, 784 validation and 1,053 test pairs. These groups come from filenames and are not confirmed independent patients. Images in the source bundle are 1024×1024; experiment-specific crop/resize settings are described in each notebook. Original `train/` and `test/` prefixes in image IDs do not define the current split.

Start with [the baseline notebook](mist-her2/baseline/baseline.ipynb). Enable a GPU, review its first configuration cell and run in order. Later experiments require additional pretrained weights or previous run outputs. Follow the [reproduction guide](docs/reproduction.md) before starting them. 

## Main observations

The baseline can produce similar IHC textures for different H&E inputs. OT and feature guidance did not consistently resolve weak conditioning or missed staining. Direct flow showed more input dependence in an inspected swap example, but often produced mask-like images. Strong HER2 signal remained difficult to reproduce. An increase in SSIM did not consistently improve DAB agreement or visual quality.

Full-test archived scores include SSIM 0.1682 / FID 186.58 for exact OT-CFM and SSIM 0.1447 / FID 367.41 for VP-CFM. The DAB-sampling results use a selected 350-image subset and must be reported separately. Published baselines use different splits/protocols, so these results do not establish a fair SOTA ranking. See [the source-linked tracking page](docs/experiment-tracking.md).

## Team

| Member | Contribution |
|---|---|
| Vasilii Gorelov | Flow Matching implementation, formulations and model development |
| Andrey Galichin | Data preparation, split pipelines and quantitative evaluation |
| Margarita Urazmetova | Experiments, sampling ablations and failure analysis |

All members contributed to literature review, interpretation, the report and presentation. Project suggested by Fedor Gubanov.

## Artifacts and reproducibility limits

Small machine-readable records are available under each experiment's `results/`. Original ZIP archives remain on disk and are excluded from Git; some exceed GitHub's regular-file size limit. Their paths, sizes and checksums are listed in [local-archives.json](docs/local-archives.json). Hosted Kaggle runs are linked in the tracking page. Checkpoints, complete prediction sets and raw image data are external dependencies. The manifest records the checksums of our local exports.

We store notebook figures and console output in `results/notebook-outputs/` to keep the source notebooks easy to browse. Full executed copies remain local and are excluded from Git. The notebooks remain Kaggle-oriented research records rather than a standalone pip package. Third-party code provenance is recorded inside the relevant notebooks; pretrained weights and datasets retain their original licenses.
