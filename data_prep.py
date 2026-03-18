import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from pipeline_utils import (
    get_logger, load_dataset,
    FRAME_RATE_S, _CONN_ROBOT_A, _CONN_ROBOT_B,
)

def compute_15_distances(targets_human: np.ndarray, targets_robot: np.ndarray) -> np.ndarray:
    """
    Calcola la distanza minima tra i 15 giunti umani e i link del robot.

    Implementazione vettorizzata: sostituisce il doppio loop Python su (H, J)
    con un'unica operazione numpy broadcast, riducendo il costo da
    H * 15 chiamate Python a un solo kernel numpy.

    Nota sulla memoria: il tensore intermedio ha shape (N, H, 15, R, 3).
    Per H=1 e R=9, N=5000 → ~16 MB (float32); per H=25 → ~400 MB.
    Se la memoria è limitata, passare un sottoinsieme di H (es. [:, 0:1, ...]).

    Args:
        targets_human: Posizioni giunti umani, shape (N, H, 15, 3).
        targets_robot:  Posizioni giunti robot,  shape (N, H, R, 3).

    Returns:
        dist_matrix: Distanza minima da ogni giunto umano a qualsiasi giunto robot,
                     shape (N, H, 15).
    """
    # (N, H, 15, 1, 3) - (N, H, 1, R, 3)  →  (N, H, 15, R, 3)
    diffs = targets_human[..., np.newaxis, :] - targets_robot[:, :, np.newaxis, ...]
    dists = np.linalg.norm(diffs, axis=-1)   # (N, H, 15, R)
    return np.min(dists, axis=-1)            # (N, H, 15)

def compute_knn_features(
    targets_h: np.ndarray,
    targets_r: np.ndarray,
    preds_h: np.ndarray,
) -> np.ndarray:
    """
    Calcola il vettore feature (N, 38) per la ricerca KNN.

    Struttura del vettore (F=38, NON scalato):
        [ dist_0 … dist_14 ]       — 15 distanze minime giunto→robot al frame H=0
        [ vel_h_0 … vel_h_14 ]     — 15 norme velocità giunti umani  (H=0 → H=1)
        [ vel_r_0 … vel_r_7  ]     —  8 norme velocità centri link robot (H=0 → H=1)

    Le velocità sono calcolate come differenze finite tra il frame H=0 e H=1
    (primo e secondo frame dell'orizzonte di predizione), divise per FRAME_RATE_S.
    Per l'umano si usano le predizioni del modello GCN (coerente tra calibrazione
    e test; disponibile in entrambi i PKL). Per il robot si usa il ground truth
    (sempre noto, zero data leakage).

    Args:
        targets_h: Posizioni GT giunti umani, shape (N, 25, 15, 3).
        targets_r: Posizioni GT link robot,   shape (N, 25,  9, 3).
        preds_h:   Predizioni GCN,            shape (N, 25, 15, 3).

    Returns:
        X_knn: Feature matrix, shape (N, 38), dtype float32.
    """
    # --- 15 distanze minime giunto→robot al frame H=0 ---
    distances = compute_15_distances(
        targets_h[:, 0:1, ...], targets_r[:, 0:1, ...]
    )[:, 0, :]                                            # (N, 15)

    # --- 15 norme velocità giunti umani (H=0 → H=1, da predizioni) ---
    vel_h_vec = (preds_h[:, 1, :, :] - preds_h[:, 0, :, :]) / FRAME_RATE_S  # (N, 15, 3)
    vel_h_norm = np.linalg.norm(vel_h_vec, axis=-1)                           # (N, 15)

    # --- 8 norme velocità centri link robot (H=0 → H=1, da GT) ---
    r_centers_0 = 0.5 * (
        targets_r[:, 0, _CONN_ROBOT_A, :] + targets_r[:, 0, _CONN_ROBOT_B, :]
    )  # (N, 8, 3)
    r_centers_1 = 0.5 * (
        targets_r[:, 1, _CONN_ROBOT_A, :] + targets_r[:, 1, _CONN_ROBOT_B, :]
    )  # (N, 8, 3)
    vel_r_norm = np.linalg.norm(
        (r_centers_1 - r_centers_0) / FRAME_RATE_S, axis=-1
    )                                                                          # (N, 8)

    return np.concatenate([distances, vel_h_norm, vel_r_norm], axis=1).astype(np.float32)


def process_offline_data(train_data_path: str, exp_dir: str, use_velocity_features: bool = True) -> dict:
    """
    Fase 1 & 2: costruisce e salva gli artefatti offline necessari alla pipeline.

    Carica il dataset di calibrazione, calcola le feature KNN (F=38: distanze +
    velocità), allena lo StandardScaler, e calcola residui e covarianze globali
    per tutti i (joint, horizon).

    Nota: il BallTree è stato rimosso. La ricerca KNN ora avviene tramite
    torch.cdist batched su GPU in score_extraction.py e test_inference.py.

    Supporta caching: se 'offline_workspace.pkl' esiste già in exp_dir, lo
    carica direttamente senza ricalcolare.

    Args:
        train_data_path: Percorso al file pickle di calibrazione
                         (es. 'chico_calib_extraction.pkl').
        exp_dir:         Directory dove salvare 'offline_workspace.pkl'.

    Returns:
        offline_artifacts: Dizionario con chiavi:
            'scaler'         — StandardScaler fittato su feature KNN (F=38).
            'storici_residui'— {j: {h: array(N, 3)}} — residui per ogni (j, h).
            'sigma_global'   — {j: {h: array(3, 3)}} — covarianza globale.
            'X_knn_scaled'   — Feature KNN scalate, shape (N, 38).
    """
    log = get_logger()
    # Workspace separato per ogni combinazione di feature: evita che un cambio
    # di use_velocity_features carichi silenziosamente il workspace sbagliato.
    # Massimo 2 file coesistenti: offline_workspace.pkl (F=15) e offline_workspace_vel.pkl (F=38).
    ws_suffix = '_vel' if use_velocity_features else ''
    ws_label  = f'38D (dist + vel)' if use_velocity_features else f'15D (solo dist)'
    log.info(f"=== FASE 1 & 2: SETUP OFFLINE (KNN feature {ws_label} + torch.cdist) ===")

    out_pkl = os.path.join(exp_dir, f'offline_workspace{ws_suffix}.pkl')
    if os.path.exists(out_pkl):
        log.info(f"Artefatti trovati in '{out_pkl}', caricamento da cache.")
        with open(out_pkl, 'rb') as f:
            return pickle.load(f)

    log.info(f"Caricamento Training Set: {train_data_path}")
    train_data = load_dataset(train_data_path)

    targets_h = train_data['targets_human']
    targets_r = train_data['targets_robot']
    preds_h = train_data['preds']

    log.info(f"Calcolo feature KNN (N, {ws_label})...")
    X_knn = compute_knn_features(targets_h, targets_r, preds_h)  # sempre (N, 38)
    if not use_velocity_features:
        # Taglia le colonne di velocità: mantieni solo le 15 distanze giunto→robot
        from pipeline_utils import N_JOINTS as _NJ
        X_knn = X_knn[:, :_NJ]   # (N, 15)

    log.info(f"Scaling a {X_knn.shape[1]} dimensioni (StandardScaler)...")
    scaler = StandardScaler().fit(X_knn)
    X_knn_scaled = scaler.transform(X_knn).astype(np.float32)

    log.info("Calcolo tensore residui e sigma globali...")
    residui_train = targets_h - preds_h  # (N, 25, 15, 3)
    N = residui_train.shape[0]

    # Struttura uniforme [j][h] — uguale a nc_scores e sigma_global:
    #   storici_residui[j][h]  →  giunto PRIMO, orizzonte SECONDO
    storici_residui = {j: {h: None for h in range(1, 26)} for j in range(15)}
    sigma_global = {j: {} for j in range(15)}

    # Vettorizzazione: calcola tutte le 375 matrici di covarianza (j, h) in un solo einsum.
    means = residui_train.mean(axis=0)              # (25, 15, 3)
    centered = residui_train - means[np.newaxis]     # (N, 25, 15, 3)
    sigma_all = np.einsum('nhjp,nhjq->hjpq', centered, centered) / (N - 1)  # (25, 15, 3, 3)

    for j in range(15):
        for h in range(1, 26):
            storici_residui[j][h] = residui_train[:, h-1, j, :]
            sigma_global[j][h] = sigma_all[h-1, j]

    offline_artifacts = {
        'scaler':          scaler,
        'storici_residui': storici_residui,
        'sigma_global':    sigma_global,
        'X_knn_scaled':    X_knn_scaled,    # (N, 38)
    }

    os.makedirs(exp_dir, exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(offline_artifacts, f)

    log.info(f"Workspace offline salvato in '{out_pkl}'.")
    return offline_artifacts