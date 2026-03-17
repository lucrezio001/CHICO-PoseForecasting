import numpy as np
import os
import pickle
import shutil
from sklearn.model_selection import train_test_split
from regressor_factory import build_regressor
from pipeline_utils import build_feature_matrix, get_logger, load_dataset, TRAIN_VAL_SPLIT

JOINT_NAMES = [
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", 
    "L_Hip", "L_Knee", "L_Ankle", "Head", "Neck", 
    "R_Shoulder", "R_Elbow", "R_Wrist", 
    "L_Shoulder", "L_Elbow", "L_Wrist"
]

# Raggruppamenti cinematici: ogni giunto condivide le feature di score passate
# con gli altri giunti della stessa catena cinematica.
# Gruppi:
#   Tronco/Bacino: [Pelvis, R_Hip, L_Hip, Head, Neck, R_Shoulder, L_Shoulder]
#   Gamba destra:  [R_Hip, R_Knee, R_Ankle]
#   Gamba sinistra:[L_Hip, L_Knee, L_Ankle]
#   Braccio destro:[R_Shoulder, R_Elbow, R_Wrist]
#   Braccio sinistro:[L_Shoulder, L_Elbow, L_Wrist]
KINEMATIC_GROUPS: dict = {}
for j in [0, 1, 4, 7, 8, 9, 12]: KINEMATIC_GROUPS[j] = [0, 1, 4, 7, 8, 9, 12]
for j in [1, 2, 3]:               KINEMATIC_GROUPS[j] = [1, 2, 3]
for j in [4, 5, 6]:               KINEMATIC_GROUPS[j] = [4, 5, 6]
for j in [9, 10, 11]:             KINEMATIC_GROUPS[j] = [9, 10, 11]
for j in [12, 13, 14]:            KINEMATIC_GROUPS[j] = [12, 13, 14]

def _prepare_val_data(config: dict, random_seed: int) -> dict:
    """
    Carica il dataset di calibrazione e ricava il validation split (5%).

    Il train_test_split usa lo stesso random_seed di tutto il training, così
    gli indici di validation sono deterministici e coerenti tra esecuzioni.

    Returns:
        Dizionario con chiavi: targets_h_val, targets_r_val, preds_h_val, val_indices.
    """
    get_logger().info(f"Preparazione dati validation set FCL ({TRAIN_VAL_SPLIT*100:.0f}% calibrazione)...")
    calib_data = load_dataset(config['directories']['val_data'])

    targets_h_full = calib_data['targets_human']
    targets_r_full = calib_data['targets_robot']
    preds_h_full   = calib_data['preds']

    indices = np.arange(targets_h_full.shape[0])
    _, val_indices = train_test_split(indices, test_size=TRAIN_VAL_SPLIT, random_state=random_seed)

    return {
        'targets_h_val': targets_h_full[val_indices],
        'targets_r_val': targets_r_full[val_indices],
        'preds_h_val':   preds_h_full[val_indices],
        'val_indices':   val_indices,
    }


def _train_or_load_models(
    config: dict,
    exp_dir: str,
    offline_artifacts: dict,
    nc_scores: dict,
    cov_diagonals: dict,
    val_indices: np.ndarray,
    sigma_global: dict,
    abl_cfg: dict,
) -> tuple[dict, dict, dict, dict, np.ndarray, np.ndarray]:
    """
    Addestra (o carica dalla cache) i 15 × 25 regressori quantile.

    Ottimizzazione rispetto all'implementazione originale:
    - Il train_test_split viene calcolato una sola volta (stessi indici per tutti
      i (joint, horizon) in quanto il seed è fisso).
    - I regressori per ogni (joint, horizon) sono indipendenti tra loro.

    Returns:
        models_dict:              {joint_idx: {h: regressor}}
        inner_val_preds:          {j: {h: np.ndarray}} — predizioni sull'inner-val
        inner_val_targets:        {j: {h: np.ndarray}} — target nc_scores sull'inner-val
        inner_val_diags:          {j: {h: np.ndarray}} — diagonali cov sull'inner-val
        val_predicted_thresholds: shape (N_val, 25, 15)
        val_covariances:          shape (N_val, 25, 15, 3, 3)
    """
    model_path   = os.path.join(exp_dir, 'trained_regressors.pkl')
    random_seed  = config['model_params']['random_state']
    temp_strat   = abl_cfg.get('temporal_strategy', 'per_horizon')
    use_distances    = abl_cfg.get('use_distances', True)
    use_time_feature = abl_cfg.get('use_time_feature', True)
    past_window      = abl_cfg.get('past_window', 2)

    X_knn_scaled = offline_artifacts['X_knn_scaled']
    N_clips      = X_knn_scaled.shape[0]
    N_val        = len(val_indices)

    pre_trained_models = None
    if os.path.exists(model_path):
        log = get_logger()
        log.info(f"Modelli già addestrati trovati in cache, caricamento da '{model_path}'.")
        with open(model_path, 'rb') as f:
            pre_trained_models = pickle.load(f)
    else:
        log = get_logger()
        log.info(f"=== FASE 3: Addestramento Regressore ({config.get('active_model', 'xgb').upper()}) ===")

    # Split deterministico calcolato UNA SOLA VOLTA — evita 375 chiamate ridondanti.
    # Dato il seed fisso, train_test_split produrrebbe sempre gli stessi indici;
    # usiamo np.random direttamente per ricavare train/val mask.
    rng = np.random.default_rng(random_seed)
    all_idx = np.arange(N_clips)
    rng.shuffle(all_idx)
    split = int(N_clips * (1.0 - TRAIN_VAL_SPLIT))
    train_idx, val_idx_inner = all_idx[:split], all_idx[split:]

    val_predicted_thresholds = np.zeros((N_val, 25, 15))
    val_covariances           = np.zeros((N_val, 25, 15, 3, 3))

    models_dict       = {}
    inner_val_preds   = {j: {} for j in range(15)}
    inner_val_targets = {j: {} for j in range(15)}
    inner_val_diags   = {j: {} for j in range(15)}

    active_model   = config.get('active_model', 'xgb').lower()
    subsample_size = config['current_run'].get('subsample_size', 0)

    if active_model == 'tabpfn' and pre_trained_models is None:
        # ── Option A: 1 modello per giunto (15 totali) ────────────────────────
        # I dati di tutti e 25 gli orizzonti vengono uniti in un unico training
        # set per giunto. L'orizzonte h è già incluso come feature scalare
        # (use_time_feature=True), quindi il modello impara la progressione
        # temporale senza bisogno di modelli separati per ogni h.
        # Speedup atteso: ~25x training, ~1.4x inference (KV cache).
        # Questa branch è SOLO per TabPFN; XGB/QRF usano il path classico sotto.
        log = get_logger()
        log.info("TabPFN Option A: addestramento 15 modelli pooled (1 per giunto)...")
        N_val_inner = len(val_idx_inner)

        for joint_idx in range(15):
            log.info(f"Addestramento giunto {joint_idx} ({JOINT_NAMES[joint_idx]})...")
            group = KINEMATIC_GROUPS[joint_idx]
            models_dict[joint_idx] = {}

            # Accumula dati da tutti e 25 gli orizzonti
            X_parts_t, Y_parts_t = [], []
            X_parts_v, Y_parts_v, D_parts_v = [], [], []
            for h in range(1, 26):
                X_h = build_feature_matrix(
                    X_knn_scaled, h, nc_scores, group,
                    use_distances, use_time_feature, past_window,
                )
                X_parts_t.append(X_h[train_idx])
                Y_parts_t.append(nc_scores[joint_idx][h][train_idx])
                X_parts_v.append(X_h[val_idx_inner])
                Y_parts_v.append(nc_scores[joint_idx][h][val_idx_inner])
                D_parts_v.append(cov_diagonals[joint_idx][h][val_idx_inner])

            X_all_t = np.vstack(X_parts_t)   # (25 * N_train, n_feat)
            Y_all_t = np.concatenate(Y_parts_t)

            # Subsample stratificato: ~subsample_size/25 campioni per orizzonte
            regressor = build_regressor(config)
            if subsample_size > 0 and X_all_t.shape[0] > subsample_size:
                rng_sub = np.random.default_rng(random_seed + joint_idx)
                n_per_h  = max(1, subsample_size // 25)
                idx_sub  = np.concatenate([
                    rng_sub.choice(np.arange(h_i * len(train_idx), (h_i + 1) * len(train_idx)),
                                   size=min(n_per_h, len(train_idx)), replace=False)
                    for h_i in range(25)
                ])
                regressor.fit(X_all_t[idx_sub], Y_all_t[idx_sub])
            else:
                regressor.fit(X_all_t, Y_all_t)

            models_dict[joint_idx]['pooled'] = regressor

            # Predici validation per tutti gli orizzonti (KV cache: 1 predict call)
            X_all_v = np.vstack(X_parts_v)   # (25 * N_val_inner, n_feat)
            preds_all_v = np.maximum(regressor.predict(X_all_v), 0)

            for h in range(1, 26):
                sl = slice((h - 1) * N_val_inner, h * N_val_inner)
                inner_val_preds[joint_idx][h]   = preds_all_v[sl]
                inner_val_targets[joint_idx][h] = Y_parts_v[h - 1]
                inner_val_diags[joint_idx][h]   = D_parts_v[h - 1]
                val_predicted_thresholds[:, h-1, joint_idx] = preds_all_v[sl]
                h_matrix = 25 if temp_strat == 'constant' else h
                val_covariances[:, h-1, joint_idx] = sigma_global[joint_idx][h_matrix]

    else:
        # ── Path classico: 375 modelli (XGB / QRF / TabPFN pre-addestrato) ───
        for joint_idx in range(15):
            if pre_trained_models is None:
                get_logger().info(f"Addestramento giunto {joint_idx} ({JOINT_NAMES[joint_idx]})...")
            else:
                get_logger().debug(f"Valutazione modello pre-esistente giunto {joint_idx}...")

            group = KINEMATIC_GROUPS[joint_idx]
            models_dict[joint_idx] = {}

            for h in range(1, 26):
                Y_h = nc_scores[joint_idx][h]
                D_h = cov_diagonals[joint_idx][h]
                X_h = build_feature_matrix(
                    X_knn_scaled, h, nc_scores, group,
                    use_distances, use_time_feature, past_window,
                )

                X_t, X_v = X_h[train_idx], X_h[val_idx_inner]
                Y_t, Y_v = Y_h[train_idx], Y_h[val_idx_inner]
                D_v      = D_h[val_idx_inner]

                if pre_trained_models is not None:
                    regressor = pre_trained_models[joint_idx][h]
                else:
                    regressor = build_regressor(config)
                    regressor.fit(X_t, Y_t)

                models_dict[joint_idx][h] = regressor

                inner_val_preds[joint_idx][h]   = np.maximum(regressor.predict(X_v), 0)
                inner_val_targets[joint_idx][h] = Y_v
                inner_val_diags[joint_idx][h]   = D_v

                val_predicted_thresholds[:, h-1, joint_idx] = inner_val_preds[joint_idx][h]
                h_matrix = 25 if temp_strat == 'constant' else h
                val_covariances[:, h-1, joint_idx] = sigma_global[joint_idx][h_matrix]

    if pre_trained_models is None:
        # TabPFN memorizza i dati di training internamente (in-context learning):
        # serializzare 375 modelli con ~4000 campioni ciascuno richiede diversi GB
        # e il file non è riutilizzabile (predict richiede i dati di contesto).
        # Per tutti gli altri modelli (XGB, QRF) il salvataggio è utile come cache.
        if active_model != 'tabpfn':
            get_logger().info(f"Salvataggio modelli addestrati in '{model_path}'...")
            with open(model_path, 'wb') as f:
                pickle.dump(models_dict, f)
            get_logger().info("Salvataggio completato.")
        else:
            get_logger().info("TabPFN: salvataggio modelli su disco saltato (in-context, non riutilizzabile).")

    return (
        models_dict,
        inner_val_preds, inner_val_targets, inner_val_diags,
        val_predicted_thresholds, val_covariances,
    )


def _generate_val_report(
    summary_path: str,
    exp_dir: str,
    inner_val_preds: dict,
    inner_val_targets: dict,
    inner_val_diags: dict,
    target_alpha: float,
) -> None:
    """
    Scrive il report testuale delle metriche di copertura sul validation set.

    I frame di riferimento sono campionati a [80, 160, 320, 400, 560, 720, 880, 1000] ms
    (frame rate 40 ms / 25 fps).

    Args:
        summary_path:      Percorso del file di output.
        exp_dir:           Directory dell'esperimento (per copiare config_used.yaml).
        inner_val_preds:   Predizioni quantile: {j: {h: array(N_inner_val)}}.
        inner_val_targets: Target nc_scores:     {j: {h: array(N_inner_val)}}.
        inner_val_diags:   Diagonali covarianza: {j: {h: array(N_inner_val, 3)}}.
        target_alpha:      Livello alpha della conformal prediction.
    """
    target_cov = (1.0 - target_alpha) * 100
    N_val = len(inner_val_preds[0][1])

    W = np.zeros((N_val, 25, 15))
    E = np.zeros((N_val, 25, 15))
    D = np.zeros((N_val, 25, 15, 3))

    for j in range(15):
        for h in range(1, 26):
            W[:, h-1, j] = inner_val_preds[j][h]
            E[:, h-1, j] = inner_val_targets[j][h]
            D[:, h-1, j] = inner_val_diags[j][h]

    mask  = E <= W
    W_mah = np.sqrt(W)

    # Conversione Mahalanobis → cm: sqrt(W) * sqrt(D) / 10  [W in mm², D in mm²]
    global_cm_matrix = (np.sqrt(W[..., None]) * np.sqrt(D)) / 10.0
    global_avg_cm    = np.mean(global_cm_matrix, axis=(0, 1, 2))

    # Frame di riferimento per le tabelle temporali (FRAME_RATE_MS = 40 ms)
    REPORT_TIMES_MS = [80, 160, 320, 400, 560, 720, 880, 1000]
    report_frames   = [(t // 40) - 1 for t in REPORT_TIMES_MS]

    with open(summary_path, 'w') as f:
        actual_cov = np.mean(mask) * 100
        mean_rad   = np.mean(W_mah)

        f.write("Conformal on CHICO (Validation Set 5%)\n\n")
        f.write("RISULTATI GLOBALI\n")
        f.write(f"Target: {target_cov:.0f}% (Alpha={target_alpha})\n")
        f.write(f"Coverage Media: {actual_cov:.2f}%\n")
        f.write(f"Mean Radius (Mahalanobis): {mean_rad:.4f}\n")
        f.write(f"Media Fisica Globale (cm) -> X: {global_avg_cm[0]:.2f} | Y: {global_avg_cm[1]:.2f} | Z: {global_avg_cm[2]:.2f}\n\n")

        f.write("DETTAGLIO TEMPORALE (Media su tutti i giunti)\n")
        f.write(f"{'Time (ms)':<10} | {'Frame':<5} | {'Coverage (%)':<12} | {'Avg Width':<10}\n")
        for ms, f_idx in zip(REPORT_TIMES_MS, report_frames):
            if f_idx < 25:
                f.write(f"{ms:<10} | {f_idx:<5} | {np.mean(mask[:, f_idx, :]) * 100:>6.2f}%       | {np.mean(W_mah[:, f_idx, :]):.4f}\n")
            else:
                f.write(f"{ms:<10} | {f_idx:<5} | N/A          | N/A\n")
        f.write("\n")

        for label, matrix, fmt in [
            ("COVERAGE (%) PER GIUNTO NEL TEMPO",        mask,  "{:>6.2f}"),
            ("AVG WIDTH (Mahalanobis) PER GIUNTO NEL TEMPO", W_mah, "{:>6.3f}"),
        ]:
            f.write(f"{label}\n")
            f.write(f"{'Giunto':<15} | " + " | ".join([f"{ms}ms" for ms in REPORT_TIMES_MS]) + "\n")
            for j in range(15):
                row_str = f"{j:02d} {JOINT_NAMES[j]:<12} |"
                for f_idx in report_frames:
                    if f_idx < 25:
                        val = np.mean(matrix[:, f_idx, j])
                        if label.startswith("COVERAGE"):
                            row_str += f" {val * 100:>6.2f} |"
                        else:
                            row_str += f" {val:>6.3f} |"
                    else:
                        row_str += "   N/A  |"
                f.write(row_str.strip(' |') + "\n")
            f.write("\n")

        f.write(f"COVERAGE E WIDTH MEDIE PER SINGOLO GIUNTO (Target {target_cov:.0f}%)\n")
        f.write(f"{'Giunto':<15} | {'Coverage (%)':<12} | {'Avg Width':<10}\n")
        for j in range(15):
            f.write(f"{j:02d} {JOINT_NAMES[j]:<12} | {np.mean(mask[:, :, j]) * 100:>6.2f}%       | {np.mean(W_mah[:, :, j]):.4f}\n")
        f.write("\n")

        f.write("DIMENSIONE INTERVALLI (± CENTIMETRI)\n\n")
        for frame_label, f_idx in [("Frame 1 (80ms)", 1), ("Frame 24 (1000ms)", 24)]:
            f.write(f"{frame_label}\n")
            f.write(f"{'Giunto':<15} | {'± X':<7} | {'± Y':<7} | {'± Z':<7}\n")
            for j in range(15):
                if f_idx < 25:
                    cm_widths = (np.sqrt(W[:, f_idx, j, None]) * np.sqrt(D[:, f_idx, j, :])) / 10.0
                    avg_cm = np.mean(cm_widths, axis=0)
                    f.write(f"{j:02d} {JOINT_NAMES[j]:<12} | {avg_cm[0]:<7.2f} | {avg_cm[1]:<7.2f} | {avg_cm[2]:<7.2f}\n")
                else:
                    f.write(f"{j:02d} {JOINT_NAMES[j]:<12} | N/A     | N/A     | N/A\n")
            f.write("\n")

    shutil.copyfile('config.yaml', os.path.join(exp_dir, 'config_used.yaml'))
    get_logger().info(f"Report metriche di validazione salvato in '{exp_dir}'.")


def train_phase3(
    config: dict,
    exp_dir: str,
    offline_artifacts: dict,
    nc_scores: dict,
    cov_diagonals: dict,
) -> tuple[dict, dict]:
    """
    Fase 3 della pipeline: addestramento dei regressori quantile e generazione
    del bundle FCL per la validazione.

    Flusso:
        1. _prepare_val_data      — carica calib .pkl e calcola gli indici di validation
        2. _train_or_load_models  — addestra o carica i 15 × 25 regressori
        3. _generate_val_report   — scrive il report testuale delle metriche

    Args:
        config:            Configurazione runtime (ablation, current_run, directories…).
        exp_dir:           Directory dell'esperimento corrente.
        offline_artifacts: Output di process_offline_data (BallTree, scaler, residui…).
        nc_scores:         Non-conformity scores: {j: {h: array(N)}}.
        cov_diagonals:     Diagonali di covarianza: {j: {h: array(N, 3)}}.

    Returns:
        models_dict:       {joint_idx: {h: regressor}} — regressori addestrati.
        val_results_bundle: Bundle dati per FCL sul validation set.
    """
    summary_path = os.path.join(exp_dir, 'val_metrics_summary.txt')
    random_seed  = config['model_params']['random_state']
    abl_cfg      = config.get('ablation', {})

    # ── 1. Validation data ────────────────────────────────────────────────────
    val_data = _prepare_val_data(config, random_seed)
    targets_h_val = val_data['targets_h_val']
    targets_r_val = val_data['targets_r_val']
    preds_h_val   = val_data['preds_h_val']
    val_indices   = val_data['val_indices']

    # ── 2. Training / loading ─────────────────────────────────────────────────
    (
        models_dict,
        inner_val_preds, inner_val_targets, inner_val_diags,
        val_predicted_thresholds, val_covariances,
    ) = _train_or_load_models(
        config, exp_dir, offline_artifacts,
        nc_scores, cov_diagonals,
        val_indices,
        offline_artifacts['sigma_global'],
        abl_cfg,
    )

    # ── 3. Report testuale ────────────────────────────────────────────────────
    if not os.path.exists(summary_path):
        get_logger().info("Generazione report metriche di validazione...")
        _generate_val_report(
            summary_path, exp_dir,
            inner_val_preds, inner_val_targets, inner_val_diags,
            target_alpha=config['current_run']['alpha'],
        )

    # ── 4. Bundle FCL ─────────────────────────────────────────────────────────
    val_results_bundle = {
        'targets_human':             targets_h_val,
        'targets_robot':             targets_r_val,
        'preds_human':               preds_h_val,
        'predicted_thresholds_mah':  val_predicted_thresholds,
        'covariances':               val_covariances,
    }

    get_logger().info("Validation bundle FCL generato.")
    return models_dict, val_results_bundle