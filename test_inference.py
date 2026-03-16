import os
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from data_prep import compute_15_distances
from train_regressor import KINEMATIC_GROUPS, JOINT_NAMES
from pipeline_utils import build_feature_matrix, KNN_BLEND_LOCAL, SVD_EPSILON, compute_mahalanobis_scores_batch, svd_pseudoinverse, get_logger

def compute_scores_for_joint(j, N_test, temp_strat, use_knn, storici_residui, sigma_global, test_residuals, test_neighbors):
    """
    Calcola i punteggi di Mahalanobis e le covarianze per il giunto j sull'intero test set.

    Strategia anti-crash (WinError 1450 su Windows con joblib):
    Restituisce array NumPy puri (3D/4D) invece di dizionari.
    Joblib li mappa automaticamente in memoria condivisa con zero overhead.

    Ottimizzazione SVD:
    - use_knn=False: una sola SVD per (j, h), Mahalanobis vettorizzato su N.
    - use_knn=True:  una SVD per campione (covarianza locale diversa per ogni n).

    Args:
        j:              Indice del giunto umano (0..14).
        N_test:         Numero di clip nel test set.
        temp_strat:     'per_horizon' oppure 'constant' (usa sempre h=25 per la sigma).
        use_knn:        Se True, usa la covarianza locale KNN per ogni campione.
        storici_residui: Residui calibrazione: storici_residui[h][j] → array(N_calib, 3).
        sigma_global:   Covarianze globali: sigma_global[j][h] → array(3, 3).
        test_residuals: Residui test set, shape (N_test, 25, 15, 3).
        test_neighbors: Indici vicini KNN per ogni clip test, shape (N_test, k) o None.

    Returns:
        j:        Indice del giunto (per riordinamento nel Parallel).
        scores_j: Punteggi Mahalanobis, shape (25, N_test).
        covs_j:   Matrici di covarianza, shape (25, N_test, 3, 3).
    """
    scores_j = np.zeros((25, N_test))
    covs_j = np.zeros((25, N_test, 3, 3))

    for h in range(1, 26):
        h_matrix = 25 if temp_strat == 'constant' else h
        res_calib = storici_residui[j][h]   # (N_calib, 3)
        sig_glob = sigma_global[j][h_matrix]
        eps_all = test_residuals[:, h-1, j, :]  # (N_test, 3)

        if not use_knn:
            # ── FAST PATH: covarianza costante per tutti i campioni
            sig_inv = svd_pseudoinverse(sig_glob, SVD_EPSILON)
            scores_j[h-1] = np.sum((eps_all @ sig_inv) * eps_all, axis=1)
            covs_j[h-1] = np.broadcast_to(sig_glob, (N_test, 3, 3)).copy()
        else:
            # ── KNN PATH: covarianza locale per ogni campione
            for n in range(N_test):
                sig_loc = np.cov(res_calib[test_neighbors[n]], rowvar=False)
                sig_target = KNN_BLEND_LOCAL * sig_loc + (1.0 - KNN_BLEND_LOCAL) * sig_glob
                sig_inv = svd_pseudoinverse(sig_target, SVD_EPSILON)
                scores_j[h-1, n] = eps_all[n] @ sig_inv @ eps_all[n]
                covs_j[h-1, n] = sig_target

    return j, scores_j, covs_j

def run_test_inference(config: dict, exp_dir: str, offline_artifacts: dict) -> tuple[str, int]:
    """
    Fase 4: inferenza batch sul test set con i regressori addestrati.

    Flusso:
    1. Carica il test set e calcola le feature di distanza KNN.
    2. Calcola i non-conformity scores per ogni (joint, horizon) in parallelo
       (joblib con prefer='threads' — numpy rilascia il GIL per le operazioni
       di algebra lineare, quindi i thread sono sufficienti e più leggeri dei processi).
    3. Esegue l'inferenza batch con i modelli addestrati per ottenere i
       predicted_thresholds per ogni clip × frame × joint.
    4. Assembla il bundle di risultati e lo salva in 'test_results.pkl'.

    Args:
        config:            Configurazione runtime.
        exp_dir:           Directory dell'esperimento (contiene 'trained_regressors.pkl').
        offline_artifacts: Output di process_offline_data.

    Returns:
        res_path:       Percorso del file 'test_results.pkl' salvato.
        N_test:         Numero di clip nel test set.
    """
    log = get_logger()
    log.info("=== FASE 4: INFERENZA SUL TEST SET (PARALLELA) ===")

    test_pkl = config['directories']['test_data']
    log.info(f"Caricamento test set: {test_pkl}")
    with open(test_pkl, 'rb') as f:
        test_data = pickle.load(f)

    targets_h = test_data['targets_human']
    targets_r = test_data['targets_robot']
    preds = test_data['preds']

    log.info("Calcolo feature spaziali (distanze KNN) sul test set...")
    distances = compute_15_distances(targets_h, targets_r)
    X_knn = distances[:, 0, :] 
    scaler = offline_artifacts['scaler']
    X_test_scaled = scaler.transform(X_knn)
    N_test = X_test_scaled.shape[0]
    
    abl_cfg = config.get('ablation', {})
    use_knn = abl_cfg.get('use_knn_matrices', True)
    temp_strat = abl_cfg.get('temporal_strategy', 'per_horizon')
    
    ball_tree = offline_artifacts['ball_tree']
    storici_residui = offline_artifacts['storici_residui']
    sigma_global = offline_artifacts['sigma_global']
    k_neighbors = config['model_params']['k_neighbors']
    
    test_neighbors = None
    if use_knn:
        log.info(f"Interrogazione BallTree per {N_test} campioni (k={k_neighbors})...")
        _, test_neighbors = ball_tree.query(X_test_scaled, k=k_neighbors)

    test_residuals = targets_h - preds

    log.info("Calcolo scores Mahalanobis (parallelo su 15 giunti)...")
    # prefer='threads': numpy/BLAS rilascia il GIL durante le operazioni di algebra lineare,
    # quindi i thread parallelizzano efficacemente senza l'overhead dei processi.
    risultati = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_scores_for_joint)(
            j, N_test, temp_strat, use_knn, storici_residui, sigma_global, test_residuals, test_neighbors
        ) for j in range(15) 
        # Ho rimosso tqdm qui perché misura la partenza e non la fine. Quando appare "Inferenza In Batch", ha finito.
    )
    
    risultati.sort(key=lambda x: x[0])
    
    # Riassembliamo i dizionari in modo che il resto del tuo codice li legga senza doverlo modificare
    true_test_scores = {j: {h: ris[1][h-1] for h in range(1, 26)} for j, ris in enumerate(risultati)}
    test_covariances = {j: {h: ris[2][h-1] for h in range(1, 26)} for j, ris in enumerate(risultati)}
                
    active_model_name = config.get('active_model', 'xgb').upper()
    log.info(f"Inferenza batch con modelli {active_model_name}...")
    model_path = os.path.join(exp_dir, 'trained_regressors.pkl')
    with open(model_path, 'rb') as f:
        models_dict = pickle.load(f)
        
    predicted_thresholds = np.zeros((N_test, 25, 15))
    use_distances = abl_cfg.get('use_distances', True)
    use_time = abl_cfg.get('use_time_feature', True)
    past_window = abl_cfg.get('past_window', 2)
    
    for j in range(15):
        group = KINEMATIC_GROUPS.get(j, [j])
        for h in range(1, 26):
            X_h = build_feature_matrix(
                X_test_scaled, h, true_test_scores, group,
                use_distances, use_time, past_window
            )
                
            regressor = models_dict[j][h]
            preds_h = np.maximum(regressor.predict(X_h), 0)
            predicted_thresholds[:, h-1, j] = preds_h

    log.info("Assemblaggio risultati finali...")
    cov_matrix_bundle = np.zeros((N_test, 25, 15, 3, 3))
    for j in range(15):
        for h in range(1, 26):
            cov_matrix_bundle[:, h-1, j, :, :] = test_covariances[j][h]
            
    results_bundle = {
        'targets_human': targets_h,
        'targets_robot': targets_r,
        'preds_human': preds,
        'predicted_thresholds_mah': predicted_thresholds,
        'covariances': cov_matrix_bundle, 
        'true_test_scores': true_test_scores
    }
    
    res_path = os.path.join(exp_dir, 'test_results.pkl')
    with open(res_path, 'wb') as f:
        pickle.dump(results_bundle, f)
        
    log.info(f"Inferenza completata. Risultati salvati in: {res_path}")
    return res_path, N_test