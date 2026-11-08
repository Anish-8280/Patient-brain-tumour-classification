# %% [markdown]
# # Patient-Independent Brain Tumour Classification from MRI
# ## using a Patient-Discounted Fisher Score (PDFS)
#
# **Pattern Recognition semester project (Monsoon 2026)**
#
# **Problem.** When slices from the same patient appear in both training and test sets, a
# classifier can recognise the *patient* instead of the *tumour*. Clinically every patient is
# new, so this notebook evaluates on **unseen patients** (patient-grouped CV and
# leave-one-patient-out, LOPO) and proposes three classical pattern-recognition changes:
#
# | | Contribution | Idea |
# |---|---|---|
# | **N1** | Patient-Discounted Fisher Score (PDFS) | Rank features by between-class variance ÷ (between-slice variance + α · between-patient variance). α = 1 is the ordinary Fisher / ANOVA-F ranking. |
# | **N2** | Patient-balanced SVM weighting | Weight each slice by 1 / (slices of its patient), so every patient counts equally. |
# | **N3** | Margin-weighted slice-to-patient fusion | Combine an unseen patient's slice scores, weighted by SVM confidence; compared with majority voting. |
# | N4 | Analysis | Patient-grouped vs ordinary inner CV for hyper-parameter tuning. |
#
# **Dataset.** Figshare brain tumour dataset (Cheng et al., 2015/2017): 3,064 T1-weighted
# contrast-enhanced slices, 233 patients, 3 classes (meningioma 708, glioma 1,426, pituitary 930).
#
# **Evaluation (instruction 10).** Patient-grouped stratified 5-fold × 3 repeats (mean ± SD) for
# all comparisons, full LOPO for the final comparison (pooled metrics, SD from a patient
# bootstrap), image-level CV only to *demonstrate* leakage. Scaling, PDFS, feature selection,
# tuning and weights are always fitted on the training patients only.
#
# **How to run:**
# 1. In Google Drive, create the folder **My Drive/figshare_brain_tumor** and upload the zip
#    from https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 (*Download all*) into it.
# 2. *Runtime → Run all*, and allow Drive access when asked. The zips are unpacked automatically.
#
# **On Kaggle instead:** *+ Add Input* → search "figshare brain tumor dataset" → add it; in the
# right panel set *Accelerator* to GPU T4 (or P100) and turn *Internet* on (for the pretrained
# ResNet-18 weights). Then *Run all*. The notebook detects Kaggle automatically; results are in
# /kaggle/working/results (Output tab).
#
# Set `QUICK = True` for a fast smoke test.
# Use a GPU runtime (*Runtime → Change runtime type → T4 GPU*) for the ResNet-18 comparison.

# %% [markdown]
# ## 0. Settings

# %%
QUICK = False            # True = small subsample, 1 repeat, tiny grids (smoke test, a few minutes)
RUN_CNN = True           # ResNet-18 comparison (comparison only; needs a GPU to be quick)
RUN_LOPO = True          # full leave-one-patient-out (E7)

# Where the data lives.
#   USE_DRIVE = True  -> put the Figshare zip in Google Drive: My Drive/figshare_brain_tumor/
#                        (survives disconnects; results are saved to My Drive/figshare_results/)
#   USE_DRIVE = False -> use Colab's temporary disk: /content/figshare_brain_tumor/
USE_DRIVE = True

import os, sys
for _s in (sys.stdout, sys.stderr):                       # Windows consoles/pipes: print ±, Δ, α safely
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
KAGGLE_INPUT = os.environ.get("PDFS_KAGGLE_INPUT", "/kaggle/input")
ON_KAGGLE = os.path.isdir(KAGGLE_INPUT) and "PDFS_DATA_DIR" not in os.environ
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))     # running as main.py
except NameError:
    _HERE = os.getcwd()                                      # running as a notebook
# The dataset ships with the repository in dataset/. When it is next to this file (local run, or
# Colab/Kaggle after `git clone`), it is used directly: no download and no Google Drive needed.
BUNDLED_DATA = os.path.join(_HERE, "dataset")
HAS_BUNDLED_DATA = os.path.isdir(os.path.join(BUNDLED_DATA, "data"))
if "PDFS_DATA_DIR" in os.environ:
    DATA_DIR = os.environ["PDFS_DATA_DIR"]
    OUT_DIR = os.environ.get("PDFS_OUT_DIR", os.path.join(_HERE, "results"))
elif HAS_BUNDLED_DATA:
    print("Using the dataset bundled with the repository")
    DATA_DIR = BUNDLED_DATA
    OUT_DIR = "/kaggle/working/results" if ON_KAGGLE else os.path.join(_HERE, "results")
elif ON_KAGGLE:
    # Kaggle: attach the dataset with "+ Add Input" (it appears under /kaggle/input, read-only);
    # it is copied to the writable working folder and unzipped there. Results: /kaggle/working/results
    print("Running on Kaggle")
    DATA_DIR = "/kaggle/working/figshare_brain_tumor"
    OUT_DIR = "/kaggle/working/results"
elif IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = "/content/drive/MyDrive/figshare_brain_tumor"
    OUT_DIR = "/content/drive/MyDrive/figshare_results"
elif IN_COLAB:
    DATA_DIR, OUT_DIR = "/content/figshare_brain_tumor", "/content/results"
else:                                                        # local run without dataset/: downloads into it
    DATA_DIR, OUT_DIR = BUNDLED_DATA, os.path.join(_HERE, "results")
print("Data folder:   ", DATA_DIR)
print("Results folder:", OUT_DIR)
SEED = 42

IMG_SIZE = 128
N_FOLDS, N_REPEATS, INNER_FOLDS = 5, 3, 3
ALPHA_SWEEP = [1, 2, 4, 8, 16, 32]       # E3 sweep of the PDFS penalty (alpha = 1 is Fisher)
ALPHA_GRID = [1, 4, 16]                  # alpha values searched when alpha is tuned
K_GRID = [200, 400, 800]                 # number of selected features searched when k is tuned
C_GRID = [0.3, 1, 3]                     # SVM C, ALWAYS tuned by inner CV. A large C gives a hard margin,
                                         # where sample weights (N2) cannot have any effect.
LOPO_ALPHA_GRID = [1, 8]                # grids for the 233-fold LOPO (tuned inside every fold)
LOPO_K_GRID = [400]
N_BOOT = 1000                            # patient bootstrap resamples for LOPO uncertainty
CNN_EPOCHS, CNN_FOLDS, CNN_SIZE = 8, 5, 224

if QUICK:
    N_REPEATS = 1
    ALPHA_SWEEP, ALPHA_GRID, K_GRID, C_GRID = [1, 8], [1, 8], [100, 400], [0.3, 1]
    LOPO_ALPHA_GRID, LOPO_K_GRID = [1, 8], [200]
    N_BOOT, CNN_EPOCHS, CNN_FOLDS = 200, 1, 1

CLASSES = ["Meningioma", "Glioma", "Pituitary"]     # Figshare labels 1, 2, 3
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

# %% [markdown]
# ## 1. Imports

# %%
import glob, hashlib, json, re, time, warnings, zipfile, subprocess
from collections import Counter

import numpy as np
import pandas as pd
import cv2
import matplotlib
try:
    get_ipython()                     # notebook: show figures inline
except NameError:
    matplotlib.use("Agg")             # script (main.py / Main.m): save figures, never block
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import (LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
rng_global = np.random.default_rng(SEED)
# Parallel workers. Each worker holds its own copy of the feature matrix, so using every core can run
# out of RAM on a 16 GB laptop (workers get killed). Capped at 6; override with PDFS_N_JOBS. Results
# do not depend on this number, only the run time does.
N_JOBS = int(os.environ.get("PDFS_N_JOBS", min(os.cpu_count() or 1, 6)))
print("CPU cores:", os.cpu_count(), "| parallel workers:", N_JOBS)

# ---- resume support: every finished experiment is saved; a re-run after a crash skips it ----
import gc, pickle
CACHE = f"{OUT_DIR}/cache"
os.makedirs(CACHE, exist_ok=True)

def _ckey(*parts):
    return hashlib.md5(repr(parts).encode()).hexdigest()[:16]

def cache_get(key):
    f = f"{CACHE}/{key}.pkl"
    if os.path.exists(f):
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None
    return None

def cache_put(key, obj):
    tmp = f"{CACHE}/{key}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, f"{CACHE}/{key}.pkl")

# %% [markdown]
# ## 2. Download the Figshare dataset
# Downloads the official archive (~880 MB) and unzips the four inner zip files of `.mat` slices.
# If the download is blocked, upload the four `brainTumorDataPublic_*.zip` files (or the `.mat`
# files) into `DATA_DIR` and re-run this cell.

# %%
def _mat_files():
    return sorted(glob.glob(f"{DATA_DIR}/**/*.mat", recursive=True),
                  key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"}

def _fetch(url, dest):
    import requests
    with requests.get(url, headers=UA, stream=True, timeout=120, allow_redirects=True) as r:
        if r.status_code != 200:
            print(f"    HTTP {r.status_code} for {url}")
            return False
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return True

def _n_slices():
    return len([p for p in _mat_files() if "cvind" not in os.path.basename(p).lower()])

def _unzip_all():
    for z in glob.glob(f"{DATA_DIR}/**/*.zip", recursive=True):
        try:
            with zipfile.ZipFile(z) as f:
                f.extractall(os.path.dirname(z))
            os.remove(z)
        except zipfile.BadZipFile:
            print("    not a valid zip:", os.path.basename(z))

def _copy_kaggle_input():
    """Copy .mat / .zip files from the read-only Kaggle input folder into DATA_DIR."""
    import shutil
    n = 0
    for src in glob.glob(f"{KAGGLE_INPUT}/**/*", recursive=True):
        if src.lower().endswith((".mat", ".zip")) and os.path.isfile(src):
            dst = os.path.join(DATA_DIR, "kaggle_input", os.path.relpath(src, KAGGLE_INPUT))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
                n += 1
    print(f"Copied {n} files from {KAGGLE_INPUT}")

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if ON_KAGGLE and _n_slices() == 0:
        _copy_kaggle_input()
    _unzip_all()                                         # handles manually uploaded zips too
    if _n_slices() > 0:
        n = _n_slices()
        print(f"Dataset present: {n} slices" + ("" if n == 3064 else "  (WARNING: the full set has 3,064)"))
        return

    # Route 1: Figshare API -> download each file separately (avoids the blocked bulk zip)
    import requests
    try:
        print("Route 1: Figshare API")
        files = requests.get("https://api.figshare.com/v2/articles/1512427/files", headers=UA, timeout=60).json()
        for f in files:
            if f["name"].endswith(".zip"):
                print(f"  downloading {f['name']} ({f['size'] / 1e6:.0f} MB)")
                _fetch(f["download_url"], f"{DATA_DIR}/{f['name']}")
        _unzip_all()
    except Exception as e:
        print("  Figshare API failed:", type(e).__name__, e)

    # Route 2: Kaggle mirror of the same Figshare data (no login needed for most accounts)
    if _n_slices() == 0:
        try:
            print("Route 2: Kaggle mirror via kagglehub")
            subprocess.run(["pip", "-q", "install", "kagglehub"], check=False)
            import kagglehub, shutil
            path = kagglehub.dataset_download("ashkhagan/figshare-brain-tumor-dataset")
            shutil.copytree(path, f"{DATA_DIR}/kaggle_mirror", dirs_exist_ok=True)
            _unzip_all()
        except Exception as e:
            print("  Kaggle mirror failed:", type(e).__name__, e)

    n = _n_slices()
    if n == 0:
        raise SystemExit(
            "Automatic download failed. Manual route:\n"
            "  1. Open https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 in your browser\n"
            "  2. Click 'Download all' (or the four brainTumorDataPublic_*.zip files)\n"
            f"  3. In Colab's Files panel, upload them into {DATA_DIR}\n"
            "  4. Run this cell again (it unzips them automatically)")
    print(f"Slices found: {n}" + ("" if n == 3064 else "  (WARNING: the full set has 3,064)"))

download_dataset()

# %% [markdown]
# ## 3. Load slices, labels and patient IDs
# Each `.mat` (MATLAB v7.3 / HDF5) holds a struct `cjdata` with `image`, `label`
# (1 = meningioma, 2 = glioma, 3 = pituitary), `PID` (patient ID) and `tumorMask`.
# The masks are **not** used: features come from the whole slice, so the method needs no
# hand-drawn tumour outline at test time.

# %%
def read_mat(path):
    try:
        import h5py
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            img = np.array(cj["image"]).T
            label = int(np.array(cj["label"]).ravel()[0])
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel() if int(c) > 0)
        return img, label, pid.strip()
    except (OSError, ImportError):
        from scipy.io import loadmat                      # older MATLAB format fallback
        cj = loadmat(path)["cjdata"][0, 0]
        pid = cj["PID"]
        pid = "".join(pid.ravel()) if pid.dtype.kind in "US" else "".join(chr(int(c)) for c in pid.ravel())
        return cj["image"], int(np.ravel(cj["label"])[0]), pid.strip()

def to_uint8(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [0.5, 99.5])                   # robust per-slice intensity scaling
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

def crop_brain(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return gray if w * h < 0.05 * gray.size else gray[y:y + h, x:x + w]

def load_one(path):
    img, label, pid = read_mat(path)
    g = to_uint8(img)
    return (cv2.resize(crop_brain(g), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA),
            label - 1, pid, hashlib.md5(g.tobytes()).hexdigest())

files = [p for p in _mat_files() if "cvind" not in os.path.basename(p).lower()]
rows = Parallel(n_jobs=N_JOBS)(delayed(load_one)(p) for p in files)
images = np.stack([r[0] for r in rows])
y = np.array([r[1] for r in rows])
pid_str = np.array([r[2] for r in rows])
md5 = [r[3] for r in rows]

# exact duplicate slices (none are expected in this dataset; checked so the report can say so)
dup = [i for i, c in Counter(md5).items() if c > 1]
keep = np.array(sorted({md5.index(h) for h in md5}))
if len(keep) < len(y):
    print(f"Removing {len(y) - len(keep)} exact duplicate slices")
    images, y, pid_str = images[keep], y[keep], pid_str[keep]

patients, groups = np.unique(pid_str, return_inverse=True)
if QUICK:                                                       # keep whole patients
    sel_p = np.concatenate([np.random.default_rng(SEED).permutation(np.unique(groups[y == c]))[:25]
                            for c in range(3)])
    m = np.isin(groups, sel_p)
    images, y, pid_str = images[m], y[m], pid_str[m]
    patients, groups = np.unique(pid_str, return_inverse=True)

spp = np.bincount(groups)
pat_label = np.array([np.bincount(y[groups == p]).argmax() for p in range(len(patients))])
mixed = sum(len(np.unique(y[groups == p])) > 1 for p in range(len(patients)))
summary = {
    "slices": int(len(y)), "patients": int(len(patients)),
    "slices_per_class": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
    "patients_per_class": {c: int((pat_label == i).sum()) for i, c in enumerate(CLASSES)},
    "slices_per_patient_min_median_max": [int(spp.min()), float(np.median(spp)), int(spp.max())],
    "patients_with_more_than_one_label": int(mixed), "exact_duplicates_removed": int(len(dup)),
}
print(json.dumps(summary, indent=2))
with open(f"{OUT_DIR}/dataset_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

fig, ax = plt.subplots(1, 4, figsize=(11, 2.8), gridspec_kw={"width_ratios": [2, 1, 1, 1]})
ax[0].hist(spp, bins=30, color="#4c72b0"); ax[0].set_xlabel("Slices per patient"); ax[0].set_ylabel("Patients")
ax[0].set_title("Patients contribute unequal numbers of slices", fontsize=9)
for c in range(3):
    ax[c + 1].imshow(images[np.nonzero(y == c)[0][0]], cmap="gray"); ax[c + 1].set_title(CLASSES[c]); ax[c + 1].axis("off")
plt.savefig(f"{OUT_DIR}/figures/dataset_overview.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 4. Handcrafted feature extraction
# Five families: first-order statistics (14), GLCM (36: 6 properties × 3 distances, mean and
# range over 4 angles), multi-scale uniform LBP (54), HOG (1,764) and Gabor (16).

# %%
GABOR = []
for f in [0.1, 0.25]:
    for t in [0, 45, 90, 135]:
        k = cv2.getGaborKernel((21, 21), 0.56 / f, np.deg2rad(t), 1 / f, 0.5, 0, ktype=cv2.CV_32F)
        GABOR.append(k - k.mean())

def f_first_order(img):
    v = img.astype(np.float32).ravel(); fg = v[v > 10] if (v > 10).sum() > 50 else v
    h = np.bincount(fg.astype(np.uint8), minlength=256) / len(fg); nz = h[h > 0]
    return np.array([fg.mean(), fg.std(), skew(fg), kurtosis(fg), -np.sum(nz * np.log2(nz)), np.sum(h ** 2),
                     fg.min(), fg.max(), *np.percentile(fg, [10, 25, 50, 75, 90]), (v > 10).mean()])

def f_glcm(img):
    q = (img.astype(np.uint16) * 32 // 256).astype(np.uint8)
    m = graycomatrix(q, [1, 3, 5], np.deg2rad([0, 45, 90, 135]), levels=32, symmetric=True, normed=True)
    out = []
    for p in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        v = graycoprops(m, p); out += [v.mean(1), np.ptp(v, 1)]
    return np.concatenate(out)

def f_lbp(img):
    return np.concatenate([np.histogram(local_binary_pattern(img, P, R, "uniform"), bins=P + 2,
                                        range=(0, P + 2), density=True)[0] for P, R in [(8, 1), (16, 2), (24, 3)]])

def f_hog(img):
    return hog(img, 9, (16, 16), (2, 2), block_norm="L2-Hys", feature_vector=True)

def f_gabor(img):
    x = img.astype(np.float32) / 255
    r = [np.abs(cv2.filter2D(x, cv2.CV_32F, k)) for k in GABOR]
    return np.array([s for a in r for s in (a.mean(), a.std())])

FAMILIES = {"First-order": f_first_order, "GLCM": f_glcm, "LBP": f_lbp, "HOG": f_hog, "Gabor": f_gabor}

t0 = time.time()
_fkey = _ckey("features", len(images), int(images.sum() % 1000003), IMG_SIZE)
fam_X = cache_get(_fkey)
if fam_X is None:
    feat_rows = Parallel(n_jobs=N_JOBS, batch_size=32)(delayed(lambda im: [fn(im) for fn in FAMILIES.values()])(im)
                                                       for im in images)
    fam_X = {name: np.nan_to_num(np.stack([r[i] for r in feat_rows]).astype(np.float32))
             for i, name in enumerate(FAMILIES)}
    cache_put(_fkey, fam_X)
    del feat_rows
else:
    print("Features loaded from cache")
X = np.hstack(list(fam_X.values()))
fam_ranges, s = {}, 0
for name, M in fam_X.items():
    fam_ranges[name] = (s, s + M.shape[1]); s += M.shape[1]
print(f"Features: {X.shape[1]} ({', '.join(f'{k}: {v.shape[1]}' for k, v in fam_X.items())}) "
      f"in {time.time() - t0:.0f}s")

# Instruction 3: dataset requirements
compliance = pd.DataFrame([
    {"Requirement (instruction 3)": "No. of attributes >= 10", "This project": f"{X.shape[1]} extracted features",
     "Met": X.shape[1] >= 10},
    {"Requirement (instruction 3)": "No. of instances >= 1000", "This project": f"{len(y)} MRI slices ({len(patients)} patients)",
     "Met": len(y) >= 1000},
    {"Requirement (instruction 3)": "Multi-category (>= 3 classes), supervised", "This project": f"{len(np.unique(y))} labelled classes",
     "Met": len(np.unique(y)) >= 3},
])
print(compliance.to_string(index=False))
if not compliance["Met"].all():
    print("NOTE: not all requirements are met on this (sub)set - expected only in QUICK mode or on partial data")

# %% [markdown]
# ## 5. The proposed methods
#
# ### N1 — Patient-Discounted Fisher Score
# For feature *j*, with class means μ_c, patient means μ_cp and N training slices:
#
# * between-class variance  **B** = Σ_c n_c (μ_c − μ)² / N
# * between-patient (within-class) variance **P** = Σ_{c,p} n_cp (μ_cp − μ_c)² / N
# * between-slice (within-patient) variance **S** = Σ (x − μ_cp)² / N
#
# **PDFS_j = B / (S + α·P)**.  Because P + S is the ordinary within-class variance, α = 1 gives
# the classical Fisher ratio (same ranking as the ANOVA F-test). α > 1 pushes down features that
# mainly encode *which patient* a slice came from. Computed on the training patients only.
#
# ### N2 — Patient-balanced weights
# Slice weight = 1 / (slices of its patient), then rescaled so every class has equal total weight.
#
# ### N3 — Margin-weighted patient fusion
# For an unseen patient: score = Σ_slices m_i · s_i, where s_i is the slice's SVM decision vector
# and m_i = (top score − second score) is its margin. Compared with majority voting.

# %%
def pdfs_components(X, y, g):
    """Variance decomposition of every column: between-class B, between-patient P, between-slice S."""
    N = len(y); mu = X.mean(0)
    B = np.zeros(X.shape[1]); P = np.zeros(X.shape[1]); S = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc, gc = X[y == c], g[y == c]
        mc = Xc.mean(0); B += len(Xc) * (mc - mu) ** 2
        for p in np.unique(gc):
            Xp = Xc[gc == p]; mp = Xp.mean(0)
            P += len(Xp) * (mp - mc) ** 2
            S += ((Xp - mp) ** 2).sum(0)
    return B / N, P / N, S / N

def pdfs_from_components(comp, alpha):
    B, P, S = comp
    return B / (S + alpha * P + 1e-12)

def pdfs_scores(X, y, g, alpha):
    """Patient-Discounted Fisher Score for every column of X (alpha = 1 -> Fisher ratio)."""
    return pdfs_from_components(pdfs_components(X, y, g), alpha)

def slice_weights(y, g, scheme):
    if scheme == "none":
        return np.ones(len(y))
    w = 1.0 / np.bincount(g)[g] if scheme == "patient" else np.ones(len(y))
    for c in np.unique(y):                                   # equal total weight per class
        w[y == c] *= 1.0 / w[y == c].sum()
    return w * len(y) / w.sum()

class PatientAwareSVM(BaseEstimator, ClassifierMixin):
    """standardise -> rank features by PDFS(alpha) -> keep top-k -> weighted RBF-SVM.
    k = None keeps every feature (no selection)."""
    def __init__(self, alpha=1.0, k=400, C=1.0, weights="class"):
        self.alpha, self.k, self.C, self.weights = alpha, k, C, weights

    def fit(self, X, y, groups):
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        keep = Z.std(0) > 1e-8
        sc = np.where(keep, pdfs_scores(Z, y, groups, self.alpha), -np.inf)
        k = keep.sum() if self.k is None else min(self.k, keep.sum())
        self.cols_ = np.argsort(-sc)[:k]
        self.svm_ = SVC(C=self.C, kernel="rbf", gamma="scale", random_state=SEED)
        self.svm_.fit(Z[:, self.cols_], y, sample_weight=slice_weights(y, groups, self.weights))
        self.classes_ = self.svm_.classes_
        return self

    def decision_function(self, X):
        return self.svm_.decision_function(self.scaler_.transform(X)[:, self.cols_])

    def predict(self, X):
        return self.svm_.predict(self.scaler_.transform(X)[:, self.cols_])

def patient_fusion(scores, g_test, method):
    """Return patient ids and fused predictions (majority vote or margin-weighted)."""
    pids = np.unique(g_test); pred = []
    for p in pids:
        s = scores[g_test == p]
        if method == "vote":
            v = np.bincount(s.argmax(1), minlength=s.shape[1]); best = np.flatnonzero(v == v.max())
            pred.append(best[np.argmax(s[:, best].sum(0))])          # ties -> summed score
        else:
            srt = np.sort(s, 1); m = srt[:, -1] - srt[:, -2]
            pred.append(np.argmax((m[:, None] * s).sum(0)))
    return pids, np.array(pred)

# %% [markdown]
# ## 6. Evaluation utilities
# Every model tunes the SVM C (and, where stated, k and α) by **patient-grouped** stratified
# inner CV on the training patients only. C must be tuned: with a large C the SVM becomes a
# hard-margin classifier on these 1,884-dimensional features, and sample weights then have no effect.
# `tune(..., grouped=False)` gives the ordinary (leaky) inner CV used in experiment E6.

# %%
def macro_f1(a, b):
    return f1_score(a, b, average="macro", labels=[0, 1, 2], zero_division=0)

def inner_splits(y, g, grouped, seed):
    if grouped:
        return list(StratifiedGroupKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(y, y, g))
    return list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(y, y))

def tune(X, y, g, alphas, ks, Cs, weights, grouped=True, seed=SEED):
    """Pick (alpha, k, C) by inner-CV macro-F1. The PDFS ranking is computed once per inner fold."""
    res = {(a, k, c): [] for a in alphas for k in ks for c in Cs}
    for tr, va in inner_splits(y, g, grouped, seed):
        scaler = StandardScaler().fit(X[tr]); Ztr, Zva = scaler.transform(X[tr]), scaler.transform(X[va])
        okc = Ztr.std(0) > 1e-8
        w = slice_weights(y[tr], g[tr], weights)
        comp = pdfs_components(Ztr, y[tr], g[tr])
        for a in alphas:
            order = np.argsort(-np.where(okc, pdfs_from_components(comp, a), -np.inf))
            for k in ks:
                cols = order[:okc.sum() if k is None else min(k, okc.sum())]
                for c in Cs:
                    m = SVC(C=c, gamma="scale").fit(Ztr[:, cols], y[tr], sample_weight=w)
                    res[(a, k, c)].append(macro_f1(y[va], m.predict(Zva[:, cols])))
    return max(res, key=lambda key: (np.mean(res[key]), -(key[1] or 10 ** 9), -key[2]))  # ties -> simpler

def run_fold(spec, Xm, y, g, tr, te):
    """Tune (alpha, k, C) on the training patients, refit, score the test patients."""
    a, k, c = tune(Xm[tr], y[tr], g[tr], spec["alphas"], spec["ks"], spec.get("Cs", C_GRID),
                   spec["weights"], grouped=spec.get("grouped", True))
    m = PatientAwareSVM(alpha=a, k=k, C=c, weights=spec["weights"]).fit(Xm[tr], y[tr], g[tr])
    sc = m.decision_function(Xm[te])
    return {"te": te, "pred": sc.argmax(1), "scores": sc, "alpha": a, "k": k, "C": c}

def run(name, spec, splits, group="", verbose=True):
    t0 = time.time()
    key = _ckey("run", name, group, spec, len(splits), len(y), C_GRID, SEED)
    r = cache_get(key)
    if r is not None:
        s = summarize(r)
        print(f"  {name:<58s} acc {s['acc']:6.2f} ± {s['acc_sd']:4.2f}  macro-F1 {s['f1']:6.2f} ± {s['f1_sd']:4.2f}   (cached)")
        return r
    Xm = Xs[spec["X"]]
    folds = Parallel(n_jobs=N_JOBS)(delayed(run_fold)(spec, Xm, y, groups, tr, te) for tr, te in splits)
    r = {"name": name, "group": group, "folds": folds, "spec": spec}
    cache_put(key, r); gc.collect()
    if verbose:
        s = summarize(r)
        print(f"  {name:<58s} acc {s['acc']:6.2f} ± {s['acc_sd']:4.2f}  macro-F1 {s['f1']:6.2f} ± {s['f1_sd']:4.2f}"
              f"   ({time.time() - t0:4.0f}s)")
    return r

def fold_stats(yt, yp):
    p, rc, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2], zero_division=0)
    return dict(acc=accuracy_score(yt, yp), f1=macro_f1(yt, yp), bal=balanced_accuracy_score(yt, yp),
                prec=p, rec=rc, f1c=f, cm=confusion_matrix(yt, yp, labels=[0, 1, 2]))

def summarize(r, level="slice", fusion="margin"):
    """Mean ± SD over CV runs (slice level, or patient level after fusion)."""
    st = []
    for f in r["folds"]:
        if level == "slice":
            st.append(fold_stats(y[f["te"]], f["pred"]))
        else:
            pids, pp = patient_fusion(f["scores"], groups[f["te"]], fusion)
            st.append(fold_stats(pat_label[pids], pp))
    arr = lambda k: np.array([s[k] for s in st]) * 100
    sd = lambda a: a.std(0, ddof=1) if len(a) > 1 else 0 * a.mean(0)
    return {"name": r["name"], "group": r["group"], "runs": len(st),
            "acc": arr("acc").mean(), "acc_sd": sd(arr("acc")), "f1": arr("f1").mean(), "f1_sd": sd(arr("f1")),
            "bal": arr("bal").mean(), "bal_sd": sd(arr("bal")),
            "prec": arr("prec").mean(0), "prec_sd": sd(arr("prec")), "rec": arr("rec").mean(0),
            "rec_sd": sd(arr("rec")), "f1c": arr("f1c").mean(0), "f1c_sd": sd(arr("f1c")),
            "cm": np.sum([s["cm"] for s in st], 0), "f1_runs": arr("f1")}

def corrected_ttest(a, b, n_train_frac=(N_FOLDS - 1) / N_FOLDS):
    """Nadeau & Bengio corrected resampled t-test on paired per-run scores."""
    d = np.asarray(a) - np.asarray(b); j = len(d)
    if j < 2 or d.var(ddof=1) == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / j + (1 - n_train_frac) / n_train_frac) * d.var(ddof=1))
    return t, 2 * stats.t.sf(abs(t), j - 1)

def table(results, level="slice", fusion="margin"):
    rows = []
    for r in results:
        s = summarize(r, level, fusion)
        rows.append({"Group": s["group"], "Method": s["name"], "Runs": s["runs"],
                     "Accuracy (%)": f"{s['acc']:.2f} ± {s['acc_sd']:.2f}",
                     "Macro-F1 (%)": f"{s['f1']:.2f} ± {s['f1_sd']:.2f}",
                     "Balanced acc. (%)": f"{s['bal']:.2f} ± {s['bal_sd']:.2f}"})
    return pd.DataFrame(rows)

def per_class(results, level="slice"):
    rows = []
    for r in results:
        s = summarize(r, level)
        for i, c in enumerate(CLASSES):
            rows.append({"Method": s["name"], "Class": c,
                         "Precision (%)": f"{s['prec'][i]:.2f} ± {s['prec_sd'][i]:.2f}",
                         "Recall (%)": f"{s['rec'][i]:.2f} ± {s['rec_sd'][i]:.2f}",
                         "F1 (%)": f"{s['f1c'][i]:.2f} ± {s['f1c_sd'][i]:.2f}"})
    return pd.DataFrame(rows)

def plot_cm(cm, title, path):
    n = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.8, 3.3)); ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{n[i, j] * 100:.1f}%\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if n[i, j] > .6 else "black")
    ax.set_xticks(range(3), CLASSES, rotation=20); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=8)
    plt.tight_layout(); plt.savefig(path); plt.show()

def save(df, name):
    df.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
    with open(f"{OUT_DIR}/{name}.md", "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for row in df.itertuples(index=False):
            fh.write("| " + " | ".join(str(v) for v in row) + " |\n")
    display(df) if "display" in globals() else print(df.to_string(index=False))

def _pipe_fold(make, Xm, y, g, tr, te, grid):
    from sklearn.model_selection import GridSearchCV
    est = make()
    if grid:
        est = GridSearchCV(est, grid, scoring="f1_macro", n_jobs=1,
                           cv=StratifiedGroupKFold(INNER_FOLDS, shuffle=True, random_state=SEED))
        est.fit(Xm[tr], y[tr], groups=g[tr])                 # tuning on training patients only
    else:
        est.fit(Xm[tr], y[tr])
    sc = est.decision_function(Xm[te]) if hasattr(est, "decision_function") else est.predict_proba(Xm[te])
    return {"te": te, "pred": est.predict(Xm[te]), "scores": np.asarray(sc, dtype=float)}

def run_pipe(name, make, Xm, splits, group, grid=None):
    """Any sklearn pipeline, fitted on the training fold only (scaling / PCA / selection inside)."""
    t0 = time.time()
    key = _ckey("pipe", name, group, grid, len(splits), len(y), SEED)
    r = cache_get(key)
    if r is None:
        folds = Parallel(n_jobs=N_JOBS)(delayed(_pipe_fold)(make, Xm, y, groups, tr, te, grid) for tr, te in splits)
        r = {"name": name, "group": group, "folds": folds}
        cache_put(key, r); gc.collect()
    s = summarize(r)
    print(f"  {name:<58s} acc {s['acc']:6.2f} ± {s['acc_sd']:4.2f}  macro-F1 {s['f1']:6.2f} ± {s['f1_sd']:4.2f}"
          f"   ({time.time() - t0:4.0f}s)")
    return r

X_raw = np.stack([cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA) for im in images]).reshape(len(y), -1).astype(np.float32)
Xs = {"all": X, **fam_X}
grouped_splits = []
for rep in range(N_REPEATS):
    grouped_splits += list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED + rep).split(X, y, groups))
image_splits = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED).split(X, y))
print(f"Patient-grouped stratified {N_FOLDS}-fold x {N_REPEATS} = {len(grouped_splits)} runs "
      f"(no patient in both training and test of any run)")
assert all(len(np.intersect1d(groups[tr], groups[te])) == 0 for tr, te in grouped_splits)

R = {}
BASE = dict(X="all", alphas=[1], ks=[400], weights="class")    # Fisher/ANOVA top-400 + class-weighted SVM (C tuned)

# %% [markdown]
# ## E1 — How much does patient leakage inflate results?
# The **same** model evaluated with image-level CV (slices of one patient can be in both
# training and test — leaky, shown only as a demonstration) and with patient-grouped CV.

# %%
print("[E1] Leakage demonstration")
R["E1_image"] = run("Image-level CV (LEAKY - demonstration only)", BASE, image_splits, "E1 leakage")
R["E1_patient"] = run("Patient-grouped CV", BASE, grouped_splits, "E1 leakage")
save(table([R["E1_image"], R["E1_patient"]]), "E1_leakage_gap")

# %% [markdown]
# ## E2 — Which feature families generalise to unseen patients?

# %%
print("[E2] Feature families (patient-grouped CV, class-weighted SVM, all features of the family)")
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
R["E2_raw"] = run_pipe("Raw pixels 64x64 + PCA(95%) (no feature extraction)",
                       lambda: Pipeline([("sc", StandardScaler()), ("pca", PCA(0.95, svd_solver="full", random_state=SEED)),
                                         ("svm", SVC(kernel="rbf", gamma="scale", class_weight="balanced"))]),
                       X_raw, grouped_splits, "E2 families", grid={"svm__C": C_GRID})
for fam in FAMILIES:
    R[f"E2_{fam}"] = run(f"{fam} only", dict(X=fam, alphas=[1], ks=[None], weights="class"), grouped_splits, "E2 families")
R["E2_all"] = run("All families, no selection", dict(X="all", alphas=[1], ks=[None], weights="class"), grouped_splits, "E2 families")
save(table([R[k] for k in R if k.startswith("E2")]), "E2_feature_families")

# %% [markdown]
# ## E3 — N1: does discounting between-patient variance help?
# Fixed k = 400 (C tuned) and a sweep over α (α = 1 is the ordinary Fisher/ANOVA ranking), then the
# version where α and k are both tuned by patient-grouped inner CV.

# %%
print("[E3] PDFS alpha sweep (k = 400, class weights, C tuned)")
for a in ALPHA_SWEEP:
    R[f"E3_a{a}"] = run(f"PDFS alpha={a}" + ("  (= Fisher / ANOVA-F)" if a == 1 else ""),
                        dict(X="all", alphas=[a], ks=[400], weights="class"), grouped_splits, "E3 PDFS")
R["E3_fisher_tuned"] = run("Fisher (alpha=1), k tuned", dict(X="all", alphas=[1], ks=K_GRID, weights="class"),
                           grouped_splits, "E3 PDFS")
R["E3_pdfs_tuned"] = run("PDFS, alpha and k tuned  [N1]", dict(X="all", alphas=ALPHA_GRID, ks=K_GRID,
                                                               weights="class"), grouped_splits, "E3 PDFS")
save(table([R[k] for k in R if k.startswith("E3")]), "E3_pdfs")

sw = [summarize(R[f"E3_a{a}"]) for a in ALPHA_SWEEP]
plt.figure(figsize=(4.6, 3))
plt.errorbar(ALPHA_SWEEP, [s["f1"] for s in sw], yerr=[s["f1_sd"] for s in sw], marker="o", capsize=3)
plt.xscale("log", base=2); plt.xlabel("α (penalty on between-patient variance)"); plt.ylabel("Macro-F1 (%)")
plt.title("PDFS on unseen patients (α = 1 is Fisher)", fontsize=9); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/E3_alpha_curve.png"); plt.show()

# Which feature families does PDFS favour? Ranking on all slices - an ILLUSTRATION of the
# selection behaviour, not an evaluation (every evaluated model selects on its training fold only).
Z = StandardScaler().fit_transform(X)
names = [f"{fam}_{i}" for fam, (a0, b0) in fam_ranges.items() for i in range(b0 - a0)]
fam_of = np.array([n.split("_")[0] for n in names])
comp_all = pdfs_components(Z, y, groups)
top = {a: np.argsort(-pdfs_from_components(comp_all, a))[:400] for a in [1, max(ALPHA_SWEEP)]}
mix = pd.DataFrame({f"Top-400 by alpha={a}": pd.Series(fam_of[top[a]]).value_counts() for a in top}).fillna(0).astype(int)
save(mix.reset_index().rename(columns={"index": "Family"}), "E3_selected_family_mix")

# %% [markdown]
# ## E4 — N2: patient-balanced weighting

# %%
print("[E4] Slice weighting (Fisher top-400, C tuned)")
for wname, label in [("none", "No weights"), ("class", "Class-balanced weights"),
                     ("patient", "Patient-balanced weights  [N2]")]:
    R[f"E4_{wname}"] = run(label, dict(X="all", alphas=[1], ks=[400], weights=wname), grouped_splits, "E4 weights")
save(table([R[k] for k in R if k.startswith("E4")]), "E4_weighting")

# %% [markdown]
# ## E5 — N3: from slices to patients
# Patient-level accuracy / macro-F1 for the same fitted models: majority vote vs margin-weighted fusion.

# %%
print("[E5] Patient-level fusion")
R["E5_proposed"] = run("PROPOSED: PDFS (tuned) + patient weights",
                       dict(X="all", alphas=ALPHA_GRID, ks=K_GRID, weights="patient"), grouped_splits, "E5")
rows = []
for key in ["E1_patient", "E5_proposed"]:
    s0 = summarize(R[key])
    rows.append({"Model": R[key]["name"], "Level": "slice", "Accuracy (%)": f"{s0['acc']:.2f} ± {s0['acc_sd']:.2f}",
                 "Macro-F1 (%)": f"{s0['f1']:.2f} ± {s0['f1_sd']:.2f}"})
    for fu, lab in [("vote", "patient - majority vote"), ("margin", "patient - margin-weighted [N3]")]:
        s1 = summarize(R[key], "patient", fu)
        rows.append({"Model": R[key]["name"], "Level": lab, "Accuracy (%)": f"{s1['acc']:.2f} ± {s1['acc_sd']:.2f}",
                     "Macro-F1 (%)": f"{s1['f1']:.2f} ± {s1['f1_sd']:.2f}"})
save(pd.DataFrame(rows), "E5_patient_fusion")

# %% [markdown]
# ## E6 — Analysis (N4): grouped vs ordinary inner tuning
# Outer evaluation is patient-grouped in both cases; only the *inner* tuning CV differs.

# %%
print("[E6] Inner tuning protocol")
for grp, lab in [(False, "Tuned with ORDINARY inner CV (leaky inner loop)"), (True, "Tuned with PATIENT-GROUPED inner CV")]:
    R[f"E6_{grp}"] = run(lab, dict(X="all", alphas=ALPHA_GRID, ks=K_GRID, weights="class", grouped=grp),
                         grouped_splits, "E6 tuning")
chosen = pd.DataFrame([{"Inner CV": "grouped" if g else "ordinary",
                        "alpha chosen": dict(Counter(f["alpha"] for f in R[f"E6_{g}"]["folds"])),
                        "k chosen": dict(Counter(f["k"] for f in R[f"E6_{g}"]["folds"])),
                        "C chosen": dict(Counter(f["C"] for f in R[f"E6_{g}"]["folds"]))} for g in [False, True]])
save(table([R["E6_False"], R["E6_True"]]), "E6_inner_tuning"); save(chosen, "E6_chosen_hyperparameters")

# %% [markdown]
# ## E7 — Final comparison: leave-one-patient-out (233 folds)
# Hyper-parameters (α, k, C) are re-tuned inside **every** LOPO fold (grouped inner CV, smaller grid).
# Each LOPO test fold is one patient (one class), so metrics are computed on the **pooled**
# predictions of all folds; the ± SD comes from resampling patients (bootstrap).
# Significance on the grouped-CV runs uses the corrected resampled t-test.

# %%
def lopo(name, spec, chunk=24):
    """LOPO in chunks of patients; progress is saved after every chunk, so a crash loses at most one chunk."""
    t0 = time.time()
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    Xm = Xs[spec["X"]]
    key = _ckey("lopo", name, spec, len(y), C_GRID, SEED)
    done = cache_get(key) or {}
    if done:
        print(f"  resuming {name}: {len(done)}/{len(splits)} patients already done")
    todo = [i for i in range(len(splits)) if i not in done]
    for c0 in range(0, len(todo), chunk):
        part = todo[c0:c0 + chunk]
        res = Parallel(n_jobs=N_JOBS)(delayed(run_fold)(spec, Xm, y, groups, *splits[i]) for i in part)
        done.update({i: {"te": r_["te"], "scores": r_["scores"]} for i, r_ in zip(part, res)})
        cache_put(key, done); del res; gc.collect()
        print(f"    {len(done)}/{len(splits)} patients ({time.time() - t0:.0f}s)")
    sc = np.zeros((len(y), 3))
    for f in done.values():
        sc[f["te"]] = f["scores"]
    print(f"  {name:<58s} done in {time.time() - t0:.0f}s")
    return {"name": name, "scores": sc, "pred": sc.argmax(1)}

def boot(res_list, n=N_BOOT, seed=SEED):
    """Patient bootstrap of pooled slice-level and patient-level macro-F1."""
    rng = np.random.default_rng(seed); P = len(patients)
    idx_by_p = [np.nonzero(groups == p)[0] for p in range(P)]
    pat_pred = {r["name"]: patient_fusion(r["scores"], groups, "margin")[1] for r in res_list}
    out = {r["name"]: {"slice": [], "patient": []} for r in res_list}
    for _ in range(n):
        ps = rng.integers(0, P, P); idx = np.concatenate([idx_by_p[p] for p in ps])
        for r in res_list:
            out[r["name"]]["slice"].append(macro_f1(y[idx], r["pred"][idx]) * 100)
            out[r["name"]]["patient"].append(macro_f1(pat_label[ps], pat_pred[r["name"]][ps]) * 100)
    return out, pat_pred

if RUN_LOPO:
    print(f"[E7] LOPO over {len(patients)} patients")
    L = [lopo("Baseline: Fisher/ANOVA (k tuned) + class weights",
              dict(X="all", alphas=[1], ks=LOPO_K_GRID, weights="class")),
         lopo("PROPOSED: PDFS (alpha, k tuned) + patient weights",
              dict(X="all", alphas=LOPO_ALPHA_GRID, ks=LOPO_K_GRID, weights="patient"))]
    bs, pat_pred = boot(L)
    rows = []
    for r in L:
        st, pidx = fold_stats(y, r["pred"]), patient_fusion(r["scores"], groups, "margin")[1]
        stp = fold_stats(pat_label, pidx)
        rows.append({"Method": r["name"], "Slice acc. (%)": f"{st['acc'] * 100:.2f}",
                     "Slice macro-F1 (%)": f"{st['f1'] * 100:.2f} ± {np.std(bs[r['name']]['slice']):.2f}",
                     "Patient acc. (%)": f"{stp['acc'] * 100:.2f}",
                     "Patient macro-F1 (%)": f"{stp['f1'] * 100:.2f} ± {np.std(bs[r['name']]['patient']):.2f}"})
        plot_cm(st["cm"], f"LOPO, slice level\n{r['name']}", f"{OUT_DIR}/figures/E7_cm_{'base' if 'Baseline' in r['name'] else 'proposed'}.png")
    save(pd.DataFrame(rows), "E7_lopo")
    d = np.array(bs[L[1]["name"]]["slice"]) - np.array(bs[L[0]["name"]]["slice"])
    dp = np.array(bs[L[1]["name"]]["patient"]) - np.array(bs[L[0]["name"]]["patient"])
    lopo_test = pd.DataFrame([{"Level": "slice", "Mean diff (pp)": f"{d.mean():+.2f}",
                               "95% CI": f"[{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]",
                               "P(diff <= 0)": f"{(d <= 0).mean():.3f}"},
                              {"Level": "patient", "Mean diff (pp)": f"{dp.mean():+.2f}",
                               "95% CI": f"[{np.percentile(dp, 2.5):+.2f}, {np.percentile(dp, 97.5):+.2f}]",
                               "P(diff <= 0)": f"{(dp <= 0).mean():.3f}"}])
    save(lopo_test, "E7_lopo_bootstrap_test")
    pc_rows = []
    for r in L:
        pr, rc, f1c, _ = precision_recall_fscore_support(y, r["pred"], labels=[0, 1, 2], zero_division=0)
        for i, c in enumerate(CLASSES):
            pc_rows.append({"Method": r["name"], "Class": c, "Precision (%)": f"{pr[i] * 100:.2f}",
                            "Recall (%)": f"{rc[i] * 100:.2f}", "F1 (%)": f"{f1c[i] * 100:.2f}"})
    save(pd.DataFrame(pc_rows), "E7_lopo_per_class")

# %% [markdown]
# ## E8 — Deep-learning comparison (ResNet-18, comparison only)
# Fine-tuned on the same patient-grouped folds (first repeat). Needs a GPU runtime to be quick.

# %%
def run_cnn():
    import torch, torch.nn as nn, torch.nn.functional as F
    from torchvision import models
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not QUICK:
        print("No GPU: running 3 folds x 3 epochs at 128 px (switch to a GPU runtime for the full comparison)")
    ep, nf, size = (CNN_EPOCHS, CNN_FOLDS, CNN_SIZE) if dev == "cuda" else (3, 3, 128)
    torch.manual_seed(SEED)
    mean = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)
    def batch(idx, aug):
        x = torch.from_numpy(images[idx]).float().div(255).unsqueeze(1).to(dev)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        if aug:
            fl = torch.rand(len(idx), device=dev) < .5; x[fl] = torch.flip(x[fl], [3])
        return (x.repeat(1, 3, 1, 1) - mean) / std
    folds = []
    for fi, (tr, te) in enumerate(grouped_splits[:nf]):
        try:
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            net = models.resnet18(weights=None)
        net.fc = nn.Linear(512, 3); net = net.to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        cw = torch.tensor(len(tr) / (3 * np.bincount(y[tr], minlength=3)), dtype=torch.float32, device=dev)
        r = np.random.default_rng(SEED + fi)
        for e in range(ep):
            net.train(); order = r.permutation(tr)
            for i in range(0, len(order), 32):
                b = order[i:i + 32]; opt.zero_grad()
                F.cross_entropy(net(batch(b, True)), torch.from_numpy(y[b]).long().to(dev), weight=cw).backward(); opt.step()
        net.eval(); sc = []
        with torch.no_grad():
            for i in range(0, len(te), 128):
                sc.append(net(batch(te[i:i + 128], False)).float().cpu().numpy())
        sc = np.concatenate(sc)
        folds.append({"te": te, "pred": sc.argmax(1), "scores": sc})
        print(f"  CNN fold {fi + 1}: macro-F1 {macro_f1(y[te], sc.argmax(1)) * 100:.2f}")
    return {"name": f"ResNet-18 fine-tuned ({nf} folds, {ep} ep, {dev})", "group": "E8 deep learning", "folds": folds}

if RUN_CNN:
    try:
        import torch  # noqa: F401
        _ck = _ckey("cnn", len(y), CNN_EPOCHS, CNN_FOLDS, CNN_SIZE, SEED)
        R["E8_cnn"] = cache_get(_ck)
        if R["E8_cnn"] is None:
            R["E8_cnn"] = run_cnn()
            cache_put(_ck, R["E8_cnn"])
        else:
            print("  CNN results loaded from cache")
        save(table([R["E8_cnn"]]), "E8_cnn")
    except ImportError:
        print("PyTorch not available - skipping E8")

# %% [markdown]
# ## E9 — Other classical classifiers on the same features
# Same patient-grouped folds; standardisation and ANOVA-F (= Fisher, α = 1) top-400 selection
# are fitted inside each training fold.

# %%
print("[E9] Classical classifier comparison (Fisher top-400 features)")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
_k9 = min(400, X.shape[1])
for nm, clf in [("kNN (k=7, distance-weighted)", lambda: KNeighborsClassifier(7, weights="distance")),
                ("Gaussian naive Bayes", lambda: GaussianNB()),
                ("LDA (shrinkage)", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                ("Random forest (300 trees)", lambda: RandomForestClassifier(300, class_weight="balanced", n_jobs=1, random_state=SEED))]:
    R[f"E9_{nm.split()[0]}"] = run_pipe(f"{nm}", lambda clf=clf: Pipeline([
        ("var", VarianceThreshold()), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=_k9)), ("clf", clf())]),
        X, grouped_splits, "E9 classifiers")
save(table([R[k] for k in R if k.startswith("E9")] + [R["E1_patient"], R["E5_proposed"]]), "E9_classifiers")

# %% [markdown]
# ## 9. Summary, significance tests and figures

# %%
key = [R["E1_patient"], R["E3_pdfs_tuned"], R["E4_patient"], R["E5_proposed"]]
summary_all = table([r for k, r in R.items() if k != "E1_image"])
save(summary_all, "summary_patient_grouped_cv")
save(per_class(key), "per_class_key_methods")

tests = []
TEST_PAIRS = [("E5_proposed", "E1_patient"), ("E3_pdfs_tuned", "E3_fisher_tuned"), ("E4_patient", "E4_class"),
              ("E6_True", "E6_False"), ("E1_patient", "E2_raw")]
TEST_RES = {}
for a, b in TEST_PAIRS:
    t, p = corrected_ttest(summarize(R[a])["f1_runs"], summarize(R[b])["f1_runs"])
    try:
        wil = stats.wilcoxon(summarize(R[a])["f1_runs"], summarize(R[b])["f1_runs"]).pvalue if N_REPEATS * N_FOLDS > 5 else np.nan
    except ValueError:                                       # identical scores in every run
        wil = 1.0
    TEST_RES[(a, b)] = (summarize(R[a])["f1"] - summarize(R[b])["f1"], p, wil)
    tests.append({"A": R[a]["name"], "B": R[b]["name"],
                  "Δ macro-F1 (pp)": f"{summarize(R[a])['f1'] - summarize(R[b])['f1']:+.2f}",
                  "p corrected t": f"{p:.4f}" if np.isfinite(p) else "-", "p Wilcoxon": f"{wil:.4f}" if np.isfinite(wil) else "-"})
save(pd.DataFrame(tests), "significance_tests")

for r in key:
    s = summarize(r)
    plot_cm(s["cm"], f"{r['name']}\n(patient-grouped CV, summed over {s['runs']} runs)",
            f"{OUT_DIR}/figures/cm_{r['group'].split()[0]}_" + re.sub(r"[^a-z0-9]+", "_", r["name"].lower())[:40].strip("_") + ".png")

ss = [summarize(r) for k, r in R.items() if k != "E1_image"]
fig, ax = plt.subplots(figsize=(7, 0.3 * len(ss) + 1))
yy = np.arange(len(ss))[::-1]
ax.barh(yy, [s["f1"] for s in ss], xerr=[s["f1_sd"] for s in ss], capsize=2,
        color=["#c44e52" if ("PROPOSED" in s["name"] or "[N" in s["name"]) else "#4c72b0" for s in ss])
ax.set_yticks(yy, [s["name"] for s in ss], fontsize=7); ax.set_xlabel("Macro-F1 (%), patient-grouped CV")
ax.set_xlim(max(0, min(s["f1"] for s in ss) - 10), 100); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figures/macro_f1_all.png"); plt.show()

with open(f"{OUT_DIR}/settings.json", "w", encoding="utf-8") as fh:
    json.dump({k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, bool, list))
               and all(isinstance(i, (int, float, str)) for i in (v if isinstance(v, list) else []))}, fh, indent=2)
print("All tables and figures saved to", OUT_DIR)

# %% [markdown]
# ## 10. Package everything
# `results.zip` contains every table (CSV + Markdown), every figure and the settings.

# %%
zip_path = f"{OUT_DIR}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p_ in glob.glob(f"{OUT_DIR}/**/*", recursive=True):
        if os.path.isfile(p_):
            z.write(p_, os.path.relpath(p_, os.path.dirname(OUT_DIR)))
print("Everything packaged in", zip_path)
try:
    from google.colab import files as _f
    _f.download(zip_path)
except Exception:
    pass
