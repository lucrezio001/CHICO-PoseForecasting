import numpy as np
import os
import pickle
from tqdm import tqdm
from pipeline_utils import KNN_BLEND_LOCAL, SVD_EPSILON, compute_mahalanobis_scores_batch, svd_pseudoinverse, get_logger

def compute_non_conformity_scores(offline_artifacts: dict, config: dict) -> tuple[dict, dict]:
    """
    Fase 2.5: calcola i non-conformity scores e le diagonali di covarianza.

    Per ogni campione n, giunto j e orizzonte h, calcola il punteggio di
    Mahalanobis del residuo rispetto alla distribuzione di calibrazione:
        score(n, j, h) = eps(n,j,h)ᵀ · Σ_inv(n,j,h) · eps(n,j,h)

    La matrice di covarianza Σ viene stimata:
    - use_knn=False: Σ = sigma_global[j][h] (costante, SVD computata una sola volta)
    - use_knn=True:  Σ = KNN_BLEND_LOCAL * Σ_locale + (1-KNN_BLEND_LOCAL) * Σ_globale

    Args:
        offline_artifacts: Output di process_offline_data.
        config:            Configurazione con 'ablation' (use_knn_matrices,
                           temporal_strategy) e 'model_params' (k_neighbors).

    Returns:
        nc_scores:     {j: {h: array(N)}}    — punteggi Mahalanobis.
        cov_diagonals: {j: {h: array(N, 3)}} — varianze per asse (X, Y, Z).
    """
    log = get_logger()
    log.info("=== FASE 2.5: Calcolo Non-Conformity Scores e Covarianze (15 giunti × 25 orizzonti) ===")
    
    abl_cfg = config.get('ablation', {})
    use_knn = abl_cfg.get('use_knn_matrices', True)
    temporal_strategy = abl_cfg.get('temporal_strategy', 'per_horizon')
    
    k_neighbors = config['model_params']['k_neighbors']
    ball_tree = offline_artifacts['ball_tree']
    X_knn_scaled = offline_artifacts['X_knn_scaled']
    storici_residui = offline_artifacts['storici_residui']
    sigma_global = offline_artifacts['sigma_global']
    
    N = X_knn_scaled.shape[0]
    
    if use_knn:
        log.info(f"Interrogazione BallTree per {N} campioni (k={k_neighbors})...")
        _, all_neighbors_indices = ball_tree.query(X_knn_scaled, k=k_neighbors)
    
    nc_scores = {j: {h: np.zeros(N) for h in range(1, 26)} for j in range(15)}
    cov_diagonals = {j: {h: np.zeros((N, 3)) for h in range(1, 26)} for j in range(15)} 
    
    for j in tqdm(range(15), desc="      Elaborazione Giunti"):
        for h in range(1, 26):
            h_matrix = 25 if temporal_strategy == 'constant' else h
            res_all = storici_residui[j][h]   # (N, 3)
            sig_glob = sigma_global[j][h_matrix]

            if not use_knn:
                # ── FAST PATH: sig_target è costante per tutti gli N campioni.
                # Una sola SVD per (j, h), poi Mahalanobis vettorizzato su tutti N.
                scores_h_j, diags_h_j = compute_mahalanobis_scores_batch(
                    res_all, sig_glob, SVD_EPSILON
                )
            else:
                # ── KNN PATH: ogni campione ha la propria covarianza locale.
                # La SVD non può essere condivisa; si calcola per ogni n.
                scores_h_j = np.zeros(N)
                diags_h_j = np.zeros((N, 3))
                for n in range(N):
                    idx_neighbors = all_neighbors_indices[n]
                    sig_loc = np.cov(res_all[idx_neighbors], rowvar=False)
                    sig_target = KNN_BLEND_LOCAL * sig_loc + (1.0 - KNN_BLEND_LOCAL) * sig_glob
                    sig_inv = svd_pseudoinverse(sig_target, SVD_EPSILON)
                    eps_n = res_all[n]
                    scores_h_j[n] = eps_n.T @ sig_inv @ eps_n
                    diags_h_j[n] = np.diag(sig_target)

            nc_scores[j][h] = scores_h_j
            cov_diagonals[j][h] = diags_h_j
            
    return nc_scores, cov_diagonals

def process_and_save_scores(offline_artifacts: dict, config: dict, offline_dir: str) -> tuple[dict, dict]:
    abl_cfg = config.get('ablation', {})
    suffix = ""
    if not abl_cfg.get('use_knn_matrices', True): suffix += "_noKNN"
    if abl_cfg.get('temporal_strategy', 'per_horizon') == 'constant': suffix += "_constTime"
    
    bundle_filename = f'scores_and_diags{suffix}.pkl'
    bundle_path = os.path.join(offline_dir, bundle_filename)
    
    log = get_logger()
    if os.path.exists(bundle_path):
        log.info(f"Bundle scores trovato in '{bundle_path}', caricamento da cache.")
        with open(bundle_path, 'rb') as f:
            data_bundle = pickle.load(f)
        return data_bundle['nc_scores'], data_bundle['cov_diagonals']

    nc_scores, cov_diagonals = compute_non_conformity_scores(offline_artifacts, config)

    log.info(f"Salvataggio bundle scores in '{bundle_path}'...")
    with open(bundle_path, 'wb') as f:
        pickle.dump({'nc_scores': nc_scores, 'cov_diagonals': cov_diagonals}, f)
        
    return nc_scores, cov_diagonals