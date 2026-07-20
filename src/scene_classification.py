# scene_classification.py
# NON-DEEP MAX for scene classification:
# RootSIFT + PCA(whiten) shared
#  - SP-VLAD (local) + IFV/Fisher Vector (local-pro) + HOG (layout) + LBP (texture) + HSV hist (color)
# Ensemble (late-fusion) with OOF CV-based weight search, then evaluate on held-out test.
#
# RUN:
#   DATA_DIR=/path/to/15_scene python src/scene_classification.py
#
# Output:
# - Prints base accuracies + ensemble acc + report + confusion matrix
# - Saves artifacts to max_fusion_artifacts.joblib and results to max_fusion_results.joblib

import os, glob, gc
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
from dataclasses import dataclass


import numpy as np
import cv2
import joblib
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


ARTIFACT_PATH = Path("max_fusion_artifacts.joblib")
RESULT_PATH   = Path("max_fusion_results.joblib")
STAGE_MODELS_PATH = Path("stage_models.joblib")
STAGE_FEATURES_PATH = Path("stage_features.joblib")


@dataclass
class Config:
    data_dir: Path = Path(os.environ.get("DATA_DIR", "data/15_scene"))
    img_exts: tuple = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    test_size: float = 0.2
    random_state: int = 42

    # ====== ACCU PRIORITY SETTINGS ======
    resize_max: int | None = 512       # 512 -> 640 often helps scene
    use_clahe: bool = True

    dsift_step: int = 8             # 6 -> 4 gives richer descriptors (slower but higher accu)
    dsift_sizes: tuple = (6, 8, 10, 12)
    max_desc_per_image: int = 1500     # cap per image to control time

    # PCA shared
    pca_dim: int = 64
    pca_whiten: bool = True
    pca_max_desc: int = 200_000
    pca_per_image: int = 1000

    # VLAD
    vlad_k: int = 64                   # 64 is strong; 128 stronger but much slower
    vlad_max_desc: int = 400_000
    vlad_per_image: int = 1500
    vlad_batch_size: int = 16384
    spm_levels: tuple = (0, 1, 2)
    intra_norm: bool = True
    power_norm: bool = True

    # IFV (Fisher Vector, diag GMM)
    ifv_k: int = 32
    ifv_max_desc: int = 300_000
    ifv_per_image: int = 1500
    ifv_reg_covar: float = 1e-4
    ifv_power_alpha: float = 0.5

    # flip augmentation
    flip_train: bool = True
    flip_train_mode: str = "avg"       # "avg" recommended (better than stacking for stability)
    flip_test: bool = True

    # Extra features
    use_hog: bool = True
    use_lbp: bool = True
    use_colorhist: bool = True

    hog_win: int = 128                 # bigger window = better layout info (slower)
    hog_cell: int = 8
    hog_block: int = 16
    hog_bins: int = 9

    lbp_grid: int = 2

    hsv_hbins: int = 24
    hsv_sbins: int = 8
    hsv_vbins: int = 8

    # SVM tuning
    grid_cv_splits: int = 5
    grid_n_jobs: int = 2              # i7 gen11 + 24GB OK

    # OOF weight search on train
    weight_grid: tuple = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    oof_splits: int = 5

    # caching
    reuse_if_exists: bool = True
    force_recompute_results: bool = True


CFG = Config()


# ----------------- data -----------------
def list_classes(data_dir: Path):
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if not classes:
        raise FileNotFoundError(f"Không thấy thư mục lớp trong: {data_dir}")
    return classes


def collect_image_paths(data_dir: Path, classes, img_exts):
    paths, labels = [], []
    class_to_id = {c: i for i, c in enumerate(classes)}
    for c in classes:
        cdir = data_dir / c
        cpaths = []
        for ext in img_exts:
            cpaths += glob.glob(str(cdir / ext))
        cpaths = sorted(cpaths)
        for p in cpaths:
            paths.append(p)
            labels.append(class_to_id[c])
    if not paths:
        raise FileNotFoundError("Không tìm thấy ảnh nào. Kiểm tra DATA_DIR/đuôi ảnh.")
    return np.array(paths), np.array(labels)


def read_gray(path: str, resize_max: int | None, use_clahe: bool):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if resize_max is not None:
        h, w = img.shape[:2]
        m = max(h, w)
        if m > resize_max:
            s = resize_max / m
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    return img


def read_bgr(path: str, resize_max: int | None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if resize_max is not None:
        h, w = img.shape[:2]
        m = max(h, w)
        if m > resize_max:
            s = resize_max / m
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ----------------- RootSIFT + Dense SIFT -----------------
def rootsift(des: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    des = des.astype(np.float32)
    des /= (np.sum(des, axis=1, keepdims=True) + eps)  # L1
    des = np.sqrt(des)
    des = normalize(des, norm="l2")
    return des.astype(np.float32)


def dense_keypoints(h: int, w: int, step: int, size: int):
    xs = np.arange(step // 2, w, step, dtype=np.float32)
    ys = np.arange(step // 2, h, step, dtype=np.float32)
    return [cv2.KeyPoint(float(x), float(y), float(size)) for y in ys for x in xs]


def extract_dense_sift(img: np.ndarray, sift, step: int, sizes: tuple, max_desc: int, rng: np.random.Generator):
    h, w = img.shape[:2]
    all_kps, all_des = [], []
    for sz in sizes:
        kps = dense_keypoints(h, w, step, sz)
        if not kps:
            continue
        kps, des = sift.compute(img, kps)
        if des is None or len(des) == 0:
            continue
        all_kps.append(kps)
        all_des.append(des)

    if not all_des:
        return None, None, (h, w)

    kps = [kp for g in all_kps for kp in g]
    des = np.vstack(all_des).astype(np.float32)

    if len(des) > max_desc:
        idx = rng.choice(len(des), max_desc, replace=False)
        des = des[idx]
        pts = np.array([kps[i].pt for i in idx], dtype=np.float32)
    else:
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)

    des = rootsift(des)
    return pts, des, (h, w)


def extract_one(path: str, sift, cfg: Config, rng: np.random.Generator, flip: bool):
    img = read_gray(path, cfg.resize_max, cfg.use_clahe)
    if img is None:
        return None, None, None
    if flip:
        img = cv2.flip(img, 1)
    pts, des, shape = extract_dense_sift(img, sift, cfg.dsift_step, cfg.dsift_sizes, cfg.max_desc_per_image, rng)
    return pts, des, shape


def sample_from_des(des: np.ndarray, take: int, rng: np.random.Generator):
    if des is None or len(des) == 0 or take <= 0:
        return None
    if len(des) <= take:
        return des
    idx = rng.choice(len(des), take, replace=False)
    return des[idx]


# ----------------- PCA (stream sampling) -----------------
def fit_pca_stream(paths: np.ndarray, sift, cfg: Config):
    rng = np.random.default_rng(cfg.random_state)
    pool, total = [], 0
    for p in tqdm(paths, desc="[1/7] PCA sampling RootSIFT", unit="img", total=len(paths)):
        _, des, _ = extract_one(p, sift, cfg, rng, flip=False)
        if des is None:
            continue
        take = min(cfg.pca_per_image, cfg.pca_max_desc - total)
        s = sample_from_des(des, take, rng)
        if s is None:
            continue
        pool.append(s)
        total += len(s)
        if total >= cfg.pca_max_desc:
            break
    if total == 0:
        raise RuntimeError("Không sample được descriptor nào để fit PCA.")
    X = np.vstack(pool).astype(np.float32)
    del pool
    gc.collect()

    pca = PCA(
        n_components=cfg.pca_dim,
        whiten=cfg.pca_whiten,
        random_state=cfg.random_state,
        svd_solver="randomized",
    )
    pca.fit(X)
    return pca


# ----------------- SPM regions -----------------
def spm_regions(levels: tuple[int, ...]):
    regs = []
    for l in levels:
        bins = 2 ** l
        for ry in range(bins):
            for rx in range(bins):
                regs.append((rx, ry, bins))
    return regs


def region_mask(pts: np.ndarray, shape: tuple[int, int], rx: int, ry: int, bins: int):
    h, w = shape
    x = pts[:, 0]
    y = pts[:, 1]
    x0, x1 = (rx / bins) * w, ((rx + 1) / bins) * w
    y0, y1 = (ry / bins) * h, ((ry + 1) / bins) * h
    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)


# ----------------- VLAD -----------------
def fit_kmeans_stream(paths: np.ndarray, sift, pca: PCA, cfg: Config):
    rng = np.random.default_rng(cfg.random_state)
    kmeans = MiniBatchKMeans(
        n_clusters=cfg.vlad_k,
        random_state=cfg.random_state,
        batch_size=cfg.vlad_batch_size,
        n_init=3,
        reassignment_ratio=0.01,
    )

    buf, buf_n, seen = [], 0, 0
    fitted_once = False

    for p in tqdm(paths, desc="[2/7] KMeans partial_fit (VLAD)", unit="img", total=len(paths)):
        _, des, _ = extract_one(p, sift, cfg, rng, flip=False)
        if des is None:
            continue
        des_p = pca.transform(des).astype(np.float32)
        take = min(cfg.vlad_per_image, cfg.vlad_max_desc - seen)
        s = sample_from_des(des_p, take, rng)
        if s is None:
            continue
        buf.append(s)
        buf_n += len(s)
        seen += len(s)

        if buf_n >= cfg.vlad_batch_size:
            B = np.vstack(buf).astype(np.float32)
            buf.clear()
            buf_n = 0
            if (not fitted_once) and len(B) < cfg.vlad_k:
                continue
            kmeans.partial_fit(B)
            fitted_once = True
            del B
            gc.collect()

        if seen >= cfg.vlad_max_desc:
            break

    if buf_n > 0:
        B = np.vstack(buf).astype(np.float32)
        if (not fitted_once) and len(B) >= cfg.vlad_k:
            kmeans.partial_fit(B)
            fitted_once = True
        elif fitted_once:
            kmeans.partial_fit(B)
        del B

    if not fitted_once:
        raise RuntimeError("KMeans chưa fit. Tăng vlad_max_desc/vlad_per_image.")
    return kmeans


def vlad_block(des: np.ndarray, kmeans: MiniBatchKMeans, cfg: Config):
    if des is None or len(des) == 0:
        return np.zeros((cfg.vlad_k, cfg.pca_dim), dtype=np.float32)

    centers = kmeans.cluster_centers_.astype(np.float32)
    words = kmeans.predict(des)
    V = np.zeros((cfg.vlad_k, des.shape[1]), dtype=np.float32)
    for k in range(cfg.vlad_k):
        m = words == k
        if not np.any(m):
            continue
        V[k] = np.sum(des[m] - centers[k], axis=0)

    if cfg.intra_norm:
        V = normalize(V, norm="l2")
    return V.astype(np.float32)


def spvlad_feature(pts, des, shape, kmeans, cfg: Config):
    regs = spm_regions(cfg.spm_levels)
    if des is None or pts is None or shape is None or len(des) == 0:
        return np.zeros(len(regs) * cfg.vlad_k * cfg.pca_dim, dtype=np.float32)
    blocks = []
    for (rx, ry, bins) in regs:
        m = region_mask(pts, shape, rx, ry, bins)
        if not np.any(m):
            V = np.zeros((cfg.vlad_k, cfg.pca_dim), dtype=np.float32)
        else:
            V = vlad_block(des[m], kmeans, cfg)
        blocks.append(V.reshape(-1))
    feat = np.concatenate(blocks, axis=0).astype(np.float32)
    if cfg.power_norm:
        feat = np.sign(feat) * np.sqrt(np.abs(feat) + 1e-12)
    feat = normalize(feat.reshape(1, -1), norm="l2").ravel().astype(np.float32)
    return feat


def build_vlad_features(paths: np.ndarray, sift, pca: PCA, kmeans: MiniBatchKMeans, cfg: Config, flip: bool):
    rng = np.random.default_rng(cfg.random_state)
    regs = spm_regions(cfg.spm_levels)
    feat_dim = len(regs) * cfg.vlad_k * cfg.pca_dim
    X = np.zeros((len(paths), feat_dim), dtype=np.float32)
    for i, p in enumerate(tqdm(paths, desc=("Build VLAD FLIP" if flip else "Build VLAD"), unit="img", total=len(paths))):
        pts, des, shape = extract_one(p, sift, cfg, rng, flip=flip)
        if des is not None and len(des) > 0:
            des = pca.transform(des).astype(np.float32)
        X[i] = spvlad_feature(pts, des, shape, kmeans, cfg)
    return X


# ----------------- IFV (Improved Fisher Vector, diag GMM) -----------------
def fit_gmm_stream(paths: np.ndarray, sift, pca: PCA, cfg: Config):
    rng = np.random.default_rng(cfg.random_state)
    pool, total = [], 0
    for p in tqdm(paths, desc="[3/7] GMM sampling (IFV)", unit="img", total=len(paths)):
        _, des, _ = extract_one(p, sift, cfg, rng, flip=False)
        if des is None:
            continue
        des_p = pca.transform(des).astype(np.float32)
        take = min(cfg.ifv_per_image, cfg.ifv_max_desc - total)
        s = sample_from_des(des_p, take, rng)
        if s is None:
            continue
        pool.append(s)
        total += len(s)
        if total >= cfg.ifv_max_desc:
            break
    if total < cfg.ifv_k:
        raise RuntimeError("Không đủ descriptors để fit GMM. Tăng ifv_max_desc/ifv_per_image.")
    X = np.vstack(pool).astype(np.float32)
    del pool
    gc.collect()

    gmm = GaussianMixture(
        n_components=cfg.ifv_k,
        covariance_type="diag",
        reg_covar=cfg.ifv_reg_covar,
        max_iter=200,
        random_state=cfg.random_state,
        init_params="kmeans",
    )
    X = X.astype(np.float64)
    gmm.fit(X)
    return gmm


def fisher_vector(des: np.ndarray, gmm: GaussianMixture, cfg: Config, eps: float = 1e-9):
    if des is None or len(des) == 0:
        return np.zeros(2 * cfg.ifv_k * cfg.pca_dim, dtype=np.float32)

    K = cfg.ifv_k
    D = des.shape[1]

    w = gmm.weights_.astype(np.float32)
    mu = gmm.means_.astype(np.float32)
    sigma = np.maximum(gmm.covariances_.astype(np.float32), eps)

    q = gmm.predict_proba(des.astype(np.float64)).astype(np.float32) # (N,K)
    N = des.shape[0]

    qT = q.T
    x = des.astype(np.float32)

    qx = qT @ x
    s0 = np.maximum(q.sum(axis=0), eps).astype(np.float32)

    sqrt_w = np.sqrt(w + eps).astype(np.float32)
    inv_sigma = 1.0 / np.sqrt(sigma)

    sum_q_diff = qx - (s0[:, None] * mu)
    u = (sum_q_diff * inv_sigma) / (N * sqrt_w[:, None] + eps)

    qx2 = qT @ (x * x)
    sum_q_xmu2 = qx2 - 2.0 * mu * qx + (s0[:, None] * (mu * mu))
    v = ((sum_q_xmu2 / sigma) - s0[:, None]) / (N * np.sqrt(2.0 * w + eps)[:, None] + eps)

    fv = np.concatenate([u.reshape(-1), v.reshape(-1)], axis=0).astype(np.float32)

    a = cfg.ifv_power_alpha
    fv = np.sign(fv) * (np.abs(fv) ** a)
    fv = fv / (np.linalg.norm(fv) + eps)
    return fv.astype(np.float32)


def build_ifv_features(paths: np.ndarray, sift, pca: PCA, gmm: GaussianMixture, cfg: Config, flip: bool):
    rng = np.random.default_rng(cfg.random_state)
    feat_dim = 2 * cfg.ifv_k * cfg.pca_dim
    X = np.zeros((len(paths), feat_dim), dtype=np.float32)
    for i, p in enumerate(tqdm(paths, desc=("Build IFV FLIP" if flip else "Build IFV"), unit="img", total=len(paths))):
        _, des, _ = extract_one(p, sift, cfg, rng, flip=flip)
        if des is not None and len(des) > 0:
            des = pca.transform(des).astype(np.float32)
        X[i] = fisher_vector(des, gmm, cfg)
    return X


# ----------------- Extra features -----------------
def hog_feature(gray: np.ndarray, cfg: Config):
    img = cv2.resize(gray, (cfg.hog_win, cfg.hog_win), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        (cfg.hog_win, cfg.hog_win),
        (cfg.hog_block, cfg.hog_block),
        (cfg.hog_cell, cfg.hog_cell),
        (cfg.hog_cell, cfg.hog_cell),
        cfg.hog_bins
    )
    feat = hog.compute(img).reshape(-1).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-12)
    return feat


def lbp_u8(gray: np.ndarray):
    g = gray.astype(np.uint8)
    center = g[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        g[0:-2, 0:-2], g[0:-2, 1:-1], g[0:-2, 2:],
        g[1:-1, 2:],   g[2:,   2:],   g[2:,   1:-1],
        g[2:,   0:-2], g[1:-1, 0:-2],
    ]
    for i, nb in enumerate(neighbors):
        codes |= ((nb >= center) << (7 - i)).astype(np.uint8)
    return codes


def lbp_feature(gray: np.ndarray, cfg: Config):
    img = cv2.resize(gray, (cfg.hog_win, cfg.hog_win), interpolation=cv2.INTER_AREA)
    codes = lbp_u8(img)
    h, w = codes.shape
    g = cfg.lbp_grid
    feats = []
    for iy in range(g):
        for ix in range(g):
            y0 = int(iy * h / g); y1 = int((iy + 1) * h / g)
            x0 = int(ix * w / g); x1 = int((ix + 1) * w / g)
            patch = codes[y0:y1, x0:x1].ravel()
            hist = np.bincount(patch, minlength=256).astype(np.float32)
            hist = hist / (hist.sum() + 1e-12)
            feats.append(hist)
    return np.concatenate(feats, axis=0).astype(np.float32)


def hsv_hist_feature(bgr: np.ndarray, cfg: Config):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [cfg.hsv_hbins, cfg.hsv_sbins, cfg.hsv_vbins],
        [0, 180, 0, 256, 0, 256]
    ).reshape(-1).astype(np.float32)
    hist = hist / (hist.sum() + 1e-12)
    hist = np.sign(hist) * np.sqrt(np.abs(hist) + 1e-12)
    hist = hist / (np.linalg.norm(hist) + 1e-12)
    return hist


def build_extra_features(paths: np.ndarray, cfg: Config, flip: bool):
    X_hog, X_lbp, X_col = [], [], []
    for p in tqdm(paths, desc=("Build EXTRA FLIP" if flip else "Build EXTRA"), unit="img", total=len(paths)):
        g = read_gray(p, cfg.resize_max, cfg.use_clahe)
        if g is None:
            if cfg.use_hog: X_hog.append(None)
            if cfg.use_lbp: X_lbp.append(None)
            if cfg.use_colorhist: X_col.append(None)
            continue

        if flip:
            g2 = cv2.flip(g, 1)
        else:
            g2 = g

        if cfg.use_hog:
            X_hog.append(hog_feature(g2, cfg))
        if cfg.use_lbp:
            X_lbp.append(lbp_feature(g2, cfg))
        if cfg.use_colorhist:
            bgr = read_bgr(p, cfg.resize_max)
            if bgr is None:
                X_col.append(None)
            else:
                if flip:
                    bgr = cv2.flip(bgr, 1)
                X_col.append(hsv_hist_feature(bgr, cfg))

    out = {}
    if cfg.use_hog:
        dim = len([x for x in X_hog if x is not None][0])
        out["hog"] = np.vstack([x if x is not None else np.zeros(dim, np.float32) for x in X_hog]).astype(np.float32)
    if cfg.use_lbp:
        dim = len([x for x in X_lbp if x is not None][0])
        out["lbp"] = np.vstack([x if x is not None else np.zeros(dim, np.float32) for x in X_lbp]).astype(np.float32)
    if cfg.use_colorhist:
        dim = len([x for x in X_col if x is not None][0])
        out["color"] = np.vstack([x if x is not None else np.zeros(dim, np.float32) for x in X_col]).astype(np.float32)
    return out


# ----------------- SVM + Scores + Ensemble -----------------
def train_linear_svc_grid(X, y, cfg: Config, title: str):
    base = LinearSVC(max_iter=40000, dual=True)
    cv = StratifiedKFold(n_splits=cfg.grid_cv_splits, shuffle=True, random_state=cfg.random_state)

    if title.lower() == "hog":
        clf = LinearSVC(C=10.0, class_weight=None, max_iter=40000, dual=True)
        clf.fit(X, y)
        return clf, {"C": 10.0, "class_weight": None}, None

    param_grid = {
        "C": [1.0, 10.0, 100.0],
        "class_weight": [None],
    }

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=cfg.grid_n_jobs,
        verbose=3,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def decision_scores(clf: LinearSVC, X):
    s = clf.decision_function(X)
    if s.ndim == 1:
        s = np.vstack([-s, s]).T
    return s.astype(np.float32)


def build_oof_scores(best_clfs: dict, X_dict: dict, y: np.ndarray, cfg: Config):
    skf = StratifiedKFold(n_splits=cfg.oof_splits, shuffle=True, random_state=cfg.random_state)
    n = len(y)
    n_classes = len(np.unique(y))
    oof = {name: np.zeros((n, n_classes), dtype=np.float32) for name in best_clfs}

    for fold, (tr, va) in enumerate(skf.split(np.zeros(n), y), start=1):
        for name, proto in best_clfs.items():
            Xtr, Xva = X_dict[name][tr], X_dict[name][va]
            ytr = y[tr]
            clf = LinearSVC(**proto.get_params())
            clf.fit(Xtr, ytr)
            oof[name][va] = decision_scores(clf, Xva)
        print(f"[OOF] fold {fold}/{cfg.oof_splits} done")
    return oof


def weight_search(oof_scores: dict, y_true: np.ndarray, weight_grid: tuple):
    names = list(oof_scores.keys())
    best_acc, best_w = -1.0, None
    grids = list(weight_grid)

    def rec(i, w_list, fused):
        nonlocal best_acc, best_w
        if i == len(names):
            y_pred = np.argmax(fused, axis=1)
            acc = (y_pred == y_true).mean()
            if acc > best_acc:
                best_acc = acc
                best_w = {names[j]: w_list[j] for j in range(len(names))}
            return
        for w in grids:
            if fused is None:
                rec(i + 1, w_list + [w], w * oof_scores[names[i]])
            else:
                rec(i + 1, w_list + [w], fused + w * oof_scores[names[i]])

    rec(0, [], None)
    return best_w, best_acc


# ----------------- MAIN -----------------
def main(cfg: Config = CFG):
    if RESULT_PATH.exists() and (not cfg.force_recompute_results):
        res = joblib.load(RESULT_PATH)
        print("Loaded saved results:")
        print("Acc(ensemble):", res["acc_ens"])
        print("Weights:", res["weights"])
        print(res["report"])
        return

    classes = list_classes(cfg.data_dir)
    paths, y = collect_image_paths(cfg.data_dir, classes, cfg.img_exts)

    X_train_p, X_test_p, y_train, y_test = train_test_split(
        paths, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    sift = cv2.SIFT_create()

    if cfg.reuse_if_exists and ARTIFACT_PATH.exists():
        art = joblib.load(ARTIFACT_PATH)
        print(f"[LOAD] {ARTIFACT_PATH}")
        pca = art["pca"]
        kmeans = art["kmeans"]
        gmm = art["gmm"]
        scalers = art["scalers"]
        clfs = art["clfs"]
        weights = art["weights"]
        cfg = art.get("cfg", cfg)
    else:
        # ========= 1) LOAD / FIT stage_models (PCA/KMeans/GMM) =========
        if STAGE_MODELS_PATH.exists():
            st = joblib.load(STAGE_MODELS_PATH)
            pca = st["pca"]
            kmeans = st["kmeans"]
            gmm = st["gmm"]
            classes = st.get("classes", classes)
            cfg = st.get("cfg", cfg)
            print("[LOAD] stage_models.joblib")
        else:
            print("\n== FIT SHARED PCA ==")
            pca = fit_pca_stream(X_train_p, sift, cfg)

            print("\n== FIT VLAD CODEBOOK ==")
            kmeans = fit_kmeans_stream(X_train_p, sift, pca, cfg)

            print("\n== FIT IFV GMM ==")
            gmm = fit_gmm_stream(X_train_p, sift, pca, cfg)

            joblib.dump(
                {"cfg": cfg, "pca": pca, "kmeans": kmeans, "gmm": gmm, "classes": classes},
                STAGE_MODELS_PATH,
                compress=3
            )
            print("[SAVE] stage_models.joblib")

        # ========= 2) LOAD / BUILD stage_features (X_dict + scalers + ytr) =========
        if STAGE_FEATURES_PATH.exists():
            stf = joblib.load(STAGE_FEATURES_PATH)
            X_dict = stf["X_dict"]
            ytr = stf["ytr"]
            scalers = stf["scalers"]
            classes = stf.get("classes", classes)
            print("[LOAD] stage_features.joblib")
        else:
            print("\n== BUILD TRAIN FEATURES ==")
            Xv = build_vlad_features(X_train_p, sift, pca, kmeans, cfg, flip=False)
            Xi = build_ifv_features(X_train_p, sift, pca, gmm, cfg, flip=False)
            ytr = y_train.copy()

            if cfg.flip_train:
                print("\n== BUILD TRAIN FEATURES (FLIP) ==")
                Xv_f = build_vlad_features(X_train_p, sift, pca, kmeans, cfg, flip=True)
                Xi_f = build_ifv_features(X_train_p, sift, pca, gmm, cfg, flip=True)

                if cfg.flip_train_mode.lower() == "stack":
                    Xv = np.vstack([Xv, Xv_f]).astype(np.float32)
                    Xi = np.vstack([Xi, Xi_f]).astype(np.float32)
                    ytr = np.hstack([y_train, y_train])
                else:
                    Xv = normalize((Xv + Xv_f), norm="l2").astype(np.float32)
                    Xi = normalize((Xi + Xi_f), norm="l2").astype(np.float32)

                del Xv_f, Xi_f
                gc.collect()

            extra = build_extra_features(X_train_p, cfg, flip=False) if (cfg.use_hog or cfg.use_lbp or cfg.use_colorhist) else {}
            if cfg.flip_train and extra:
                extra_f = build_extra_features(X_train_p, cfg, flip=True)
                if cfg.flip_train_mode.lower() == "stack":
                    for k in extra:
                        extra[k] = np.vstack([extra[k], extra_f[k]]).astype(np.float32)
                else:
                    for k in extra:
                        extra[k] = normalize((extra[k] + extra_f[k]), norm="l2").astype(np.float32)
                del extra_f
                gc.collect()

            scalers = {}
            X_dict = {"vlad": Xv, "ifv": Xi}

            if extra:
                for name, Xf in extra.items():
                    sc = StandardScaler(with_mean=True, with_std=True)
                    X_dict[name] = sc.fit_transform(Xf).astype(np.float32)
                    scalers[name] = sc

            joblib.dump(
                {"X_dict": X_dict, "ytr": ytr, "scalers": scalers, "classes": classes},
                STAGE_FEATURES_PATH,
                compress=3
            )
            print("[SAVE] stage_features.joblib")

          

        # ========= 3) TRAIN models + weights =========
        print("\n== TRAIN BASE MODELS (GRIDSEARCH) ==")
        clfs = {}
        for name, Xf in X_dict.items():
            print(f"\n--- {name.upper()} ---")
            clf, bp, bcv = train_linear_svc_grid(Xf, ytr, cfg, title=name)
            clfs[name] = {"est": clf, "best_params": bp, "best_cv": bcv}
            print(f"[{name}] best_params={bp} best_cv={bcv}")

        print("\n== OOF WEIGHT SEARCH (TRAIN) ==")
        best_clfs = {name: clfs[name]["est"] for name in clfs}
        oof_scores = build_oof_scores(best_clfs, X_dict, ytr, cfg)
        weights, oof_acc = weight_search(oof_scores, ytr, cfg.weight_grid)
        print("Best weights (OOF):", weights, "OOF-acc:", oof_acc)

        joblib.dump({
            "classes": classes,
            "cfg": cfg,
            "pca": pca,
            "kmeans": kmeans,
            "gmm": gmm,
            "scalers": scalers,
            "clfs": clfs,
            "weights": weights,
        }, ARTIFACT_PATH)
        print(f"[SAVE] {ARTIFACT_PATH}")

        
    # ===== TEST =====
    print("\n== BUILD TEST FEATURES ==")
    Xv_te = build_vlad_features(X_test_p, sift, pca, kmeans, cfg, flip=False)
    Xi_te = build_ifv_features(X_test_p, sift, pca, gmm, cfg, flip=False)

    if cfg.flip_test:
        Xv_te_f = build_vlad_features(X_test_p, sift, pca, kmeans, cfg, flip=True)
        Xi_te_f = build_ifv_features(X_test_p, sift, pca, gmm, cfg, flip=True)
        Xv_te = normalize((Xv_te + Xv_te_f), norm="l2").astype(np.float32)
        Xi_te = normalize((Xi_te + Xi_te_f), norm="l2").astype(np.float32)
        del Xv_te_f, Xi_te_f
        gc.collect()

    X_te = {"vlad": Xv_te, "ifv": Xi_te}

    extra_te = build_extra_features(X_test_p, cfg, flip=False) if (cfg.use_hog or cfg.use_lbp or cfg.use_colorhist) else {}
    if cfg.flip_test and extra_te:
        extra_te_f = build_extra_features(X_test_p, cfg, flip=True)
        for k in extra_te:
            extra_te[k] = normalize((extra_te[k] + extra_te_f[k]), norm="l2").astype(np.float32)
        del extra_te_f
        gc.collect()

    if extra_te:
        for name, Xf in extra_te.items():
            sc = scalers.get(name, None)
            if sc is None:
                continue
            X_te[name] = sc.transform(Xf).astype(np.float32)

    print("\n== BASE ACCURACIES ==")
    scores = {}
    for name, Xf in X_te.items():
        if name not in clfs:
            continue
        clf = clfs[name]["est"]
        yp = clf.predict(Xf)
        acc = accuracy_score(y_test, yp)
        print(f"{name.upper():>6} acc = {acc:.4f}")
        scores[name] = decision_scores(clf, Xf)

    fused = None
    for name, sc in scores.items():
        w = float(weights.get(name, 0.0))
        fused = sc * w if fused is None else (fused + sc * w)

    y_pred_ens = np.argmax(fused, axis=1)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    cm = confusion_matrix(y_test, y_pred_ens)
    report = classification_report(y_test, y_pred_ens, target_names=classes)

    print("\n== ENSEMBLE (MAX) ==")
    print("Weights:", weights)
    print("Acc(ensemble):", acc_ens)
    print("\nReport:\n", report)
    print("\nConfusion matrix:\n", cm)

    joblib.dump({
        "acc_ens": acc_ens,
        "weights": weights,
        "cm": cm,
        "report": report,
        "classes": classes,
    }, RESULT_PATH)
    print(f"[SAVE] {RESULT_PATH}")


if __name__ == "__main__":
    main(CFG)
