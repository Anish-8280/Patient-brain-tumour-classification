# Patient-Independent Brain Tumour Classification from MRI Using Handcrafted Features

Pattern Recognition semester project — IIIT Sri City, Monsoon 2026
**Team:** Anish Mahapatra · Saumya Suman · Aditya Ranjan

Classifies brain MRI slices as **meningioma, glioma or pituitary tumour**, and evaluates every method on **patients the
model has never seen**. Image-level splits put slices of the same patient in both training and test sets, which
inflates accuracy. This project measures that effect and tests classical pattern-recognition methods under honest,
patient-wise evaluation.

---

## Key results

Patient-grouped stratified 5-fold CV × 3 repeats (15 runs), macro-F1 (%, mean ± SD):

| Method | Macro-F1 |
|---|---|
| Baseline, **image-level CV** (leaky, for comparison only) | 88.31 ± 1.77 |
| Baseline: Fisher top-400 features + RBF-SVM | **80.82 ± 4.26** |
| Raw pixels + PCA (no feature extraction) | 77.05 ± 2.72 |
| PDFS feature selection (proposed) | 80.08 ± 4.33 |
| Patient-balanced SVM weights (proposed) | 80.07 ± 4.31 |
| Baseline + patient-level margin-weighted fusion | **85.70 ± 5.56** |
| ResNet-18 fine-tuned (deep learning, comparison only, 5 runs) | 91.83 ± 3.13 |

**Findings**
- Patient leakage inflates macro-F1 by **7.5 points** (88.31 → 80.82).
- Handcrafted features beat raw pixels by **3.8 points**; LBP is the strongest single family.
- **Combining an unseen patient's slices** raises macro-F1 to **85.7%**.
- The proposed **PDFS** and **patient-balanced weighting** did **not** improve on the Fisher baseline. This negative
  result is confirmed by leave-one-patient-out (LOPO) evaluation and analysed in the report.
- Meningioma is the hardest class: the main error is glioma predicted as meningioma.

---

## Method

```
MRI slice → crop brain, resize 128×128 → 1,884 handcrafted features
         → standardise + Fisher / PDFS ranking, top-k → RBF-SVM (class / patient weights)
         → per-slice prediction → margin-weighted fusion → per-patient diagnosis
```

| Feature family | Dim. |
|---|---|
| First-order intensity statistics | 14 |
| GLCM (Haralick), 3 distances × 4 angles | 36 |
| Multi-scale uniform LBP | 54 |
| HOG | 1,764 |
| Gabor filter bank | 16 |

**Patient-Discounted Fisher Score (PDFS).** Each feature's within-class variance is split into a between-patient part
*P* and a within-patient part *S*. Features are ranked by

PDFS(α) = B / (S + α·P), where B is the between-class variance.

α = 1 gives the ordinary Fisher score; α > 1 penalises features that mainly identify the patient.

---

## Experiments

| | Experiment |
|---|---|
| E1 | Leakage gap: image-level vs. patient-grouped CV |
| E2 | Raw pixels vs. each feature family vs. all features |
| E3 | PDFS α sweep; tuned PDFS vs. tuned Fisher |
| E4 | No / class-balanced / patient-balanced SVM weights |
| E5 | PDFS + patient weights; slice- vs. patient-level fusion |
| E6 | Patient-grouped vs. ordinary inner tuning |
| E7 | Leave-one-patient-out (233 folds) with patient bootstrap |
| E8 | ResNet-18 (deep learning, comparison only) |
| E9 | k-NN, naive Bayes, LDA, random forest on the same features |

**Evaluation**
- Patient-grouped **stratified** 5-fold CV, repeated 3 times; LOPO for the final comparison.
- **No leakage:** scaling, feature ranking and selection, PCA, sample weights and all hyperparameters (C, k, α) are
  fitted on training patients only, with nested patient-grouped tuning.
- Accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrices, mean ± SD.
- Corrected resampled t-test (Nadeau & Bengio) and Wilcoxon signed-rank test.

---

## Dataset

[Figshare brain tumour dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) (Cheng et al.):
3,064 T1-weighted contrast-enhanced slices from **233 patients**, with a patient ID on every slice.

| Class | Slices | Patients |
|---|---|---|
| Meningioma | 708 | 82 |
| Glioma | 1,426 | 89 |
| Pituitary | 930 | 62 |

The full dataset is included in this repository under `dataset/` (3,064 `.mat` slices plus `cvind.mat`), so
`main.py` runs offline with no download. If the `dataset/` folder is missing, the code downloads the data into it
automatically; if that download is blocked, get the zip from Figshare or the
[Kaggle mirror](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset) and put it in `dataset/`;
it is unzipped automatically.

---

## How to run

**MATLAB** (as required for evaluation):
```matlab
Main
```
Runs the whole study through `main.py`, prints every table and displays every figure.

**Python:**
```bash
pip install -r requirements.txt
python main.py
```

**Google Colab / Kaggle:** open `PDFS_Brain_Tumor_LOPO.ipynb` and choose *Run all*. On Kaggle, add the
*Figshare Brain Tumor Dataset* through **+ Add Input**. A GPU is used automatically for ResNet-18.

| Setting (top of `main.py`) | Effect |
|---|---|
| `QUICK = True` | Small subset, 1 repeat: a smoke test in a few minutes |
| `RUN_LOPO = False` | Skip leave-one-patient-out (saves about 45 min) |
| `RUN_CNN = False` | Skip the ResNet-18 comparison |

**Runtime:** about 1.5–2 h on 4 CPU cores plus a GPU. If a run crashes, run it again: finished experiments are cached
in `results/cache/`, so it resumes where it stopped.

---

## Outputs

Everything is written to `results/` and zipped as `results.zip`:

| File | Contents |
|---|---|
| `summary_patient_grouped_cv.csv/.md` | All methods: accuracy, macro-F1, balanced accuracy |
| `E1_…` – `E9_…` | One table per experiment |
| `per_class_key_methods.csv/.md` | Per-class precision, recall, F1 |
| `E7_lopo*.csv/.md` | LOPO results and bootstrap test |
| `significance_tests.csv/.md` | Paired tests |
| `figures/` | Confusion matrices, α curve, comparison chart, dataset overview |
| `dataset_summary.json`, `settings.json` | Data counts and the exact settings used |

---

## Repository structure

```
├── Main.m                        # MATLAB entry point (runs main.py, shows results)
├── main.py                       # complete pipeline: data → features → E1–E9 → results
├── PDFS_Brain_Tumor_LOPO.ipynb   # same code as a Colab / Kaggle notebook
├── requirements.txt
└── dataset/                      # Figshare dataset, included (no download needed)
```

---

## Team contributions

| Member | Contribution |
|---|---|
| Anish Mahapatra | Data loading, patient-wise splits, evaluation framework; E1, E6, E7 |
| Saumya Suman | Feature extraction, PDFS, significance tests; E2, E3 |
| Aditya Ranjan | Patient weighting, patient fusion, classifiers, CNN; E4, E5, E8, E9 |

---

## References

- J. Cheng *et al.*, "Enhanced performance of brain tumor classification via tumor region augmentation and partition," *PLoS ONE*, 2015.
- R. O. Duda, P. E. Hart, D. G. Stork, *Pattern Classification*, 2nd ed., Wiley, 2001.
- C. Nadeau, Y. Bengio, "Inference for the generalization error," *Machine Learning*, 2003.
