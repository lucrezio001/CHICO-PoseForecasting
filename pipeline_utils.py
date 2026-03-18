"""
Utility condivise per la pipeline di conformal prediction CHICO.
Contiene costanti, helper per feature engineering, operazioni SVD vettorizzate
e helper di I/O (logging strutturato, caricamento dataset).
"""
import logging
import os
import pickle
import numpy as np

# ─── COSTANTI PIPELINE ────────────────────────────────────────────────────────

N_JOINTS: int = 15        # Numero di giunti umani nel dataset CHICO
N_HORIZONS: int = 25      # Frame futuri predetti (1–25)
FRAME_RATE_MS: int = 40   # Intervallo tra frame in millisecondi (25 fps)
FRAME_RATE_S: float = FRAME_RATE_MS / 1000.0  # Intervallo tra frame in secondi (0.04 s)

# Connettività link robot: 9 giunti sequenziali → 8 link (base → end-effector).
# Identica a CONN_ROBOT in evaluate_fcl.py; ridefinita qui per evitare import circolare.
N_ROBOT_LINKS: int = 8
_CONN_ROBOT_A: np.ndarray = np.arange(N_ROBOT_LINKS, dtype=np.int32)      # (8,) start joint
_CONN_ROBOT_B: np.ndarray = np.arange(1, N_ROBOT_LINKS + 1, dtype=np.int32)  # (8,) end joint

# Dimensionalità del vettore feature KNN (usato da BallTree / torch.cdist):
#   15 distanze minime giunto→robot  (stato spaziale)
#   15 norme velocità giunti umani   (cinematica umana, H=0→H=1)
#    8 norme velocità link robot     (cinematica robot, H=0→H=1)
N_KNN_FEATURES: int = N_JOINTS + N_JOINTS + N_ROBOT_LINKS  # = 38

# Frazione riservata al validation set interno (split deterministico sul set di calibrazione).
# Il 95% va al training del regressore, il 5% alla valutazione FCL.
TRAIN_VAL_SPLIT: float = 0.05

# Coefficiente di mixing per la stima della covarianza locale (KNN blend)
# sig_target = KNN_BLEND_LOCAL * sig_locale + (1 - KNN_BLEND_LOCAL) * sig_globale
KNN_BLEND_LOCAL: float = 0.95

# Soglia di regolarizzazione numerica per la pseudoinversa SVD:
# autovalori < SVD_EPSILON vengono annullati invece di invertiti.
SVD_EPSILON: float = 0.001


# ─── LOGGING ──────────────────────────────────────────────────────────────────

def setup_logging(exp_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configura e restituisce il logger principale della pipeline.

    Scrive su stdout (sempre) e su file 'pipeline.log' dentro exp_dir (se fornito).
    Il formato include timestamp, livello e messaggio — utile per batch notturni.

    Args:
        exp_dir: Directory dell'esperimento dove salvare 'pipeline.log'. Se None,
                 scrive solo su stdout.
        level:   Livello di logging (default: INFO). Usa logging.DEBUG per debug.

    Returns:
        logger: Logger configurato con nome 'chico'.
    """
    logger = logging.getLogger('chico')
    # Evita handler duplicati se chiamato più volte (es. batch con N esperimenti)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if exp_dir is not None:
        os.makedirs(exp_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(exp_dir, 'pipeline.log'))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Restituisce il logger 'chico' già configurato (o un logger base se non ancora setup)."""
    return logging.getLogger('chico')


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_dataset(pkl_path: str) -> dict:
    """
    Carica un dataset pickle e verifica le chiavi obbligatorie.

    Args:
        pkl_path: Percorso al file .pkl con chiavi
                  'targets_human', 'targets_robot', 'preds'.

    Returns:
        Dizionario con i dati del dataset.

    Raises:
        FileNotFoundError: se il file non esiste.
        KeyError: se mancano chiavi obbligatorie.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Dataset non trovato: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    required = {'targets_human', 'targets_robot', 'preds'}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Chiavi mancanti nel dataset '{pkl_path}': {missing}")
    return data


def validate_offline_artifacts(artifacts: dict) -> None:
    """
    Verifica che offline_artifacts contenga tutte le chiavi obbligatorie
    e che la dimensione delle feature sia coerente con N_KNN_FEATURES.

    Nota: 'ball_tree' NON è più una chiave richiesta (rimpiazzato da torch.cdist).
    Se un workspace vecchio contiene ancora 'ball_tree' viene ignorato silenziosamente.
    Se X_knn_scaled ha 15 colonne (vecchio formato) viene sollevato ValueError chiaro
    per forzare la rigenerazione del workspace con F=38.

    Args:
        artifacts: Dizionario prodotto da process_offline_data().

    Raises:
        KeyError:   Se mancano chiavi obbligatorie.
        ValueError: Se la shape di X_knn_scaled non corrisponde a N_KNN_FEATURES.
    """
    required = {'scaler', 'storici_residui', 'sigma_global', 'X_knn_scaled'}
    missing = required - set(artifacts.keys())
    if missing:
        raise KeyError(f"offline_artifacts: chiavi mancanti {missing}")
    n_feat = artifacts['X_knn_scaled'].shape[1]
    if n_feat != N_KNN_FEATURES:
        raise ValueError(
            f"X_knn_scaled ha {n_feat} colonne, atteso N_KNN_FEATURES={N_KNN_FEATURES} "
            f"(15 dist + 15 vel_umano + 8 vel_robot). "
            "Elimina 'offline_workspace.pkl' per rigenerare il workspace con le nuove feature."
        )


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def knn_torch_batched(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    k: int,
    batch_size: int = 8000,
) -> np.ndarray:
    """
    Ricerca k-NN esatta tramite torch.cdist batched su GPU (fallback CPU).

    Sostituisce sklearn BallTree per trovare i k vicini più prossimi in norma L2.
    Vantaggi rispetto a BallTree:
      - GPU: ~15ms per 60k query contro 30k ref (F=38) vs ~60ms/query×N con BallTree
      - Nessuna struttura ausiliaria da costruire e serializzare
      - Scala linearmente con N_ref, ottimale per k/N >= 5% (BallTree degenera)

    Memoria per chunk (GPU, batch_size=8000, N_ref=30k):
      8000 × 30000 × 4 byte = 960 MB — sicuro con 16GB VRAM

    Args:
        X_query:    Feature di query, shape (N_query, F). Numpy float32/64.
        X_ref:      Feature di riferimento (calibration set), shape (N_ref, F).
        k:          Numero di vicini da restituire.
        batch_size: Query per iterazione GPU. Riduci se OOM (default 8000).

    Returns:
        indices: Indici dei k vicini più prossimi per ogni query,
                 shape (N_query, k), dtype int64.
    """
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_ref_t = torch.tensor(X_ref, dtype=torch.float32, device=device)
    N = len(X_query)
    indices = np.empty((N, k), dtype=np.int64)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X_q_t = torch.tensor(X_query[start:end], dtype=torch.float32, device=device)
        D = torch.cdist(X_q_t, X_ref_t)                          # (batch, N_ref)
        _, idx = torch.topk(D, k=k, largest=False, dim=1)        # k smallest
        indices[start:end] = idx.cpu().numpy()

    del X_ref_t
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass

    return indices


def build_feature_matrix(
    X_scaled: np.ndarray,
    h: int,
    scores: dict,
    group: list,
    use_distances: bool,
    use_time_feature: bool,
    past_window: int,
) -> np.ndarray:
    """
    Costruisce la matrice delle feature per il regressore al passo temporale h.

    Questa funzione centralizza la logica di feature engineering identica
    presente in train_regressor.py e test_inference.py.

    Args:
        X_scaled:        Feature KNN scalate, shape (N, N_KNN_FEATURES=38).
                         Contiene 15 distanze + 15 vel umano + 8 vel robot (scalate).
        h:               Orizzonte temporale corrente (1-indexed, range 1..25).
        scores:          Dizionario di non-conformity scores: scores[j][h] → array(N).
        group:           Lista degli indici di giunto nel gruppo cinematico del giunto corrente.
        use_distances:   Se True, include X_scaled come feature.
        use_time_feature: Se True, aggiunge h come feature scalare.
        past_window:     Numero di passi passati da includere come contesto (0 = nessuno).

    Returns:
        X_h: Matrice delle feature, shape (N, n_features).
    """
    N = X_scaled.shape[0]

    X_h = X_scaled if use_distances else np.empty((N, 0))

    if use_time_feature:
        X_h = np.hstack([X_h, np.full((N, 1), h)])

    if past_window > 0:
        context_features = []
        for pw in range(1, past_window + 1):
            past_h = h - pw
            if past_h >= 1:
                group_scores = np.column_stack([scores[g][past_h] for g in group])
            else:
                group_scores = np.zeros((N, len(group)))
            context_features.append(group_scores)
        X_h = np.hstack([X_h] + context_features)

    return X_h


# ─── OPERAZIONI SVD ───────────────────────────────────────────────────────────

def svd_pseudoinverse(sig: np.ndarray, threshold: float = SVD_EPSILON) -> np.ndarray:
    """
    Calcola la pseudoinversa di una matrice di covarianza 3×3 tramite SVD.

    Gli autovalori < threshold vengono annullati per stabilità numerica.

    Args:
        sig:       Matrice di covarianza, shape (3, 3).
        threshold: Soglia di regolarizzazione per gli autovalori.

    Returns:
        sig_inv: Pseudoinversa, shape (3, 3).
    """
    U, S, Vh = np.linalg.svd(sig)
    S_inv = np.where(S >= threshold, 1.0 / S, 0.0)
    return Vh.T @ np.diag(S_inv) @ U.T


def compute_mahalanobis_scores_batch(
    residuals: np.ndarray,
    sig: np.ndarray,
    threshold: float = SVD_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola i punteggi di Mahalanobis e le diagonali di covarianza in modo vettorizzato,
    ottimizzato per il caso in cui sig è costante per tutti i campioni (use_knn=False).

    Esegue una sola SVD per la matrice sig invece di N SVD ripetute all'interno
    del loop su n, riducendo il costo computazionale da O(N * 3³) a O(3³) + O(N * 3).

    Args:
        residuals: Residui, shape (N, 3).
        sig:       Matrice di covarianza condivisa, shape (3, 3).
        threshold: Soglia di regolarizzazione SVD.

    Returns:
        scores: Punteggi di Mahalanobis, shape (N,).
        diags:  Diagonali della matrice di covarianza, shape (N, 3).
                (Identiche per tutti gli N in questo caso costante.)
    """
    sig_inv = svd_pseudoinverse(sig, threshold)
    # eps @ sig_inv: (N, 3) @ (3, 3) = (N, 3)
    # somma elemento per elemento con eps: riduzione per riga → (N,)
    scores = np.sum((residuals @ sig_inv) * residuals, axis=1)
    diags = np.tile(np.diag(sig), (residuals.shape[0], 1))
    return scores, diags
