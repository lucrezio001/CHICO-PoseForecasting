import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from pipeline_utils import get_logger

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

def process_offline_data(train_data_path: str, exp_dir: str) -> dict:
    """
    Fase 1 & 2: costruisce e salva gli artefatti offline necessari alla pipeline.

    Carica il dataset di calibrazione, calcola le distanze giunto-robot al
    frame presente (H=0), allena lo StandardScaler e il BallTree, e calcola
    residui e covarianze globali per tutti i (joint, horizon).

    Supporta caching: se 'offline_workspace.pkl' esiste già in exp_dir, lo
    carica direttamente senza ricalcolare.

    Args:
        train_data_path: Percorso al file pickle di calibrazione
                         (es. 'chico_calib_extraction.pkl').
        exp_dir:         Directory dove salvare 'offline_workspace.pkl'.

    Returns:
        offline_artifacts: Dizionario con chiavi:
            'scaler'         — StandardScaler fittato su distanze di calibrazione.
            'ball_tree'      — BallTree per ricerca KNN.
            'storici_residui'— {j: {h: array(N, 3)}} — residui per ogni (j, h).
            'sigma_global'   — {j: {h: array(3, 3)}} — covarianza globale.
            'X_knn_scaled'   — Feature di distanza scalate, shape (N, 15).
    """
    log = get_logger()
    log.info("=== FASE 1 & 2: SETUP OFFLINE (BALLTREE SU DISTANZE) ===")

    out_pkl = os.path.join(exp_dir, 'offline_workspace.pkl')
    if os.path.exists(out_pkl):
        log.info(f"Artefatti trovati in '{out_pkl}', caricamento da cache.")
        with open(out_pkl, 'rb') as f:
            return pickle.load(f)

    log.info(f"Caricamento Training Set: {train_data_path}")
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)

    targets_h = train_data['targets_human']
    targets_r = train_data['targets_robot']
    preds_h = train_data['preds']

    log.info("Estrazione distanze (N, 15) al frame presente (H=0)...")
    # Calcoliamo le distanze solo all'orizzonte 0 (presente)
    S_train = compute_15_distances(targets_h[:, 0:1, ...], targets_r[:, 0:1, ...])[:, 0, :]

    log.info("Scaling a 15 dimensioni (StandardScaler)...")
    scaler = StandardScaler().fit(S_train)
    S_train_scaled = scaler.transform(S_train).astype(np.float32)

    log.info("Addestramento BallTree...")
    ball_tree = BallTree(S_train_scaled)

    log.info("Calcolo tensore residui e sigma globali...")
    residui_train = targets_h - preds_h  # (N, 25, 15, 3)
    
    # Struttura uniforme [j][h] — uguale a nc_scores e sigma_global:
    #   storici_residui[j][h]  →  giunto PRIMO, orizzonte SECONDO
    storici_residui = {j: {h: None for h in range(1, 26)} for j in range(15)}
    sigma_global = {j: {} for j in range(15)}

    for j in range(15):
        for h in range(1, 26):
            res_h = residui_train[:, h-1, j, :]
            storici_residui[j][h] = res_h
            sigma_global[j][h] = np.cov(res_h, rowvar=False)

    offline_artifacts = {
        'scaler': scaler, 
        'ball_tree': ball_tree,
        'storici_residui': storici_residui, 
        'sigma_global': sigma_global,
        'X_knn_scaled': S_train_scaled
    }
    
    os.makedirs(exp_dir, exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(offline_artifacts, f)
        
    log.info(f"Workspace offline salvato in '{out_pkl}'.")
    return offline_artifacts