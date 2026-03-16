import os
import pickle
import numpy as np
import fcl
from tqdm import tqdm
from joblib import Parallel, delayed
from pipeline_utils import get_logger

SOGLIA_COLLISIONE = 130.0  # mm — distanza minima corpo-robot sotto la quale si considera collisione

HUMAN_ATLAS_RADII = {
    "head": 0.10, "torso": 0.18, "upper_arm": 0.06, 
    "lower_arm": 0.05, "upper_leg": 0.08, "lower_leg": 0.07,
}

# Connessioni scheletriche umane (coppie di indici giunto che definiscono i segmenti ossei).
# Indici seguono la convenzione CHICO-15: 0=Pelvis, 1=R_Hip, 2=R_Knee, 3=R_Ankle,
# 4=L_Hip, 5=L_Knee, 6=L_Ankle, 7=Head, 8=Neck, 9=R_Shoulder, 10=R_Elbow,
# 11=R_Wrist, 12=L_Shoulder, 13=L_Elbow, 14=L_Wrist.
CONN_HUMAN = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (4, 12), (0, 8), (12, 8), (8, 7),
    (12, 13), (13, 14), (11, 10), (10, 9), (8, 9), (9, 1)
]
# Connessioni robot: link sequenziali da base (0) a end-effector (8).
CONN_ROBOT = [(i, i+1) for i in range(8)]
# Raggio approssimato dei link del robot collaborativo (mm).
# Valore conservativo empirico per cobot industriali a braccio circolare (~4 cm).
RAGGIO_ROBOT = 40.0  # mm

def get_human_radius(joint_a, joint_b):
    """Ritorna il raggio anatomico in millimetri per un segmento osseo."""
    joint_mapping = {
        "head": [(8, 7)],
        "torso": [(0, 1), (0, 4), (0, 8), (12, 8), (8, 9), (4, 12), (9, 1)],
        "upper_arm": [(12, 13), (10, 9)], 
        "lower_arm": [(13, 14), (11, 10)],
        "upper_leg": [(1, 2), (4, 5)], 
        "lower_leg": [(2, 3), (5, 6)] 
    }
    part_name = "torso"
    for part, connections in joint_mapping.items():
        if (joint_a, joint_b) in connections or (joint_b, joint_a) in connections:
            part_name = part
            break
    return HUMAN_ATLAS_RADII[part_name] * 1000.0

def get_joint_radius(joint_idx):
    """Calcola il raggio anatomico massimo tra i segmenti collegati a un giunto."""
    raggi = []
    for (a, b) in CONN_HUMAN:
        if joint_idx == a or joint_idx == b:
            raggi.append(get_human_radius(a, b))
    if not raggi:
        return 50.0 
    return max(raggi)

# Lookup precomputato: JOINT_RADII[j] = raggio anatomico massimo per il giunto j (mm).
# Evita di ricalcolare get_joint_radius() all'interno dei loop per ogni frame/clip.
JOINT_RADII = {j: get_joint_radius(j) for j in range(15)}


def crea_cilindro(p1, p2, raggio):
    """Crea un cilindro fisico (per Robot e Ground Truth)."""
    v_diff = p2 - p1
    lunghezza = np.linalg.norm(v_diff)
    if lunghezza < 1e-6:
        return fcl.CollisionObject(fcl.Cylinder(raggio, 0.001), fcl.Transform(np.eye(3), p1))
    direzione = v_diff / lunghezza
    centro = (p1 + p2) / 2.0
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, direzione)
    c = np.dot(z_axis, direzione)
    if c < -0.9999: 
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    else:
        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + v_skew + np.dot(v_skew, v_skew) * (1 / (1 + c))
    return fcl.CollisionObject(fcl.Cylinder(raggio, lunghezza), fcl.Transform(R, centro))

def crea_ellissoide_statistico(centro, cov_matrix, s_hat):
    """Crea un ellissoide puro orientato secondo gli autovettori della covarianza."""
    U, S, Vh = np.linalg.svd(cov_matrix)
    
    # Semi-assi: radice(soglia * autovalore). Aggiungiamo 1e-3 per stabilità di FCL
    a = np.sqrt(s_hat * S[0]) + 1e-3
    b = np.sqrt(s_hat * S[1]) + 1e-3
    c = np.sqrt(s_hat * S[2]) + 1e-3
    
    # Correzione del determinante (FCL esplode se non è una rotazione pura)
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
        
    return fcl.CollisionObject(fcl.Ellipsoid(a, b, c), fcl.Transform(U, centro))

def check_collision_gt(pose_umano, pose_robot):
    """GROUND TRUTH: Segmenti ossei (Cilindri) vs Robot (Cilindri)."""
    num_frames = pose_umano.shape[0]
    req, res = fcl.DistanceRequest(), fcl.DistanceResult()
    for t in range(num_frames):
        robot_objs = [crea_cilindro(pose_robot[t, a], pose_robot[t, b], RAGGIO_ROBOT) for a, b in CONN_ROBOT]
        for (a, b) in CONN_HUMAN:
            raggio_anatomico = get_human_radius(a, b)
            human_obj = crea_cilindro(pose_umano[t, a], pose_umano[t, b], raggio_anatomico)
            for r_obj in robot_objs:
                if fcl.distance(human_obj, r_obj, req, res) <= SOGLIA_COLLISIONE:
                    return True 
    return False

def check_collision_pred_conformal(pose_umano_pred, pose_robot, W_clip, Cov_clip):
    """PREDIZIONE: Giunti (Ellissoidi Statistici) vs Robot (Cilindri)."""
    num_frames = pose_umano_pred.shape[0]
    req, res = fcl.DistanceRequest(), fcl.DistanceResult()
    for t in range(num_frames):
        robot_objs = [crea_cilindro(pose_robot[t, a], pose_robot[t, b], RAGGIO_ROBOT) for a, b in CONN_ROBOT]
        
        for j in range(15):
            s_hat = max(W_clip[t, j], 0.0)
            cov_matrix = Cov_clip[t, j]
            raggio_anatomico_giunto = JOINT_RADII[j]  # precomputato a livello modulo
            
            # Oggetto geometrico: Incertezza pura orientata
            human_obj = crea_ellissoide_statistico(pose_umano_pred[t, j], cov_matrix, s_hat)
            
            # Margine dinamico: 13cm fissi + la "carne" dell'operatore attorno al giunto
            soglia_dinamica = SOGLIA_COLLISIONE + raggio_anatomico_giunto
            
            for r_obj in robot_objs:
                if fcl.distance(human_obj, r_obj, req, res) <= soglia_dinamica:
                    return True 
    return False

def check_collision_dynamic_clipping(preds_h_frame, preds_h_prev, targets_r_frame, W_frame, Cov_frame, m_base, tau):
    """
    Implementazione esatta:
    Se Conformal lancia allarme (> 13cm + r_ant), allarghiamo il raggio del cilindro.
    Il crash è definito dalla distanza < 130mm tra i volumi.
    """
    m_base_mm = m_base * 1000.0  # conversione metri → millimetri
    
    # 1. Stadio 1: Check Conformal Prediction
    # check_collision_pred_conformal internamente usa già la logica della distanza
    allarme_conformal = check_collision_pred_conformal(
        preds_h_frame[np.newaxis, ...], 
        targets_r_frame[np.newaxis, ...], 
        W_frame[np.newaxis, ...], 
        Cov_frame[np.newaxis, ...]
    )

    # 2. Calcolo velocità per raggio extra
    vel_human = np.zeros(15)
    if preds_h_prev is not None:
        vel_human = np.linalg.norm(preds_h_frame - preds_h_prev, axis=1) / 0.04

    robot_objs = []
    for a, b in CONN_ROBOT:
        robot_objs.append(crea_cilindro(targets_r_frame[a], targets_r_frame[b], RAGGIO_ROBOT))

    # 3. Controllo collisioni frame-by-frame
    for a, b in CONN_HUMAN:
        r_ant = get_human_radius(a, b)  # raggio per segmento osseo (non giunto singolo)
        
        # Se c'è allarme conformal, aggiungiamo il raggio extra dello pseudo-codice
        if allarme_conformal:
            v_seg = max(vel_human[a], vel_human[b])
            r_extra = m_base_mm + (tau * v_seg)
            r_test = r_ant + r_extra
        else:
            # Altrimenti usiamo il raggio base (Baseline)
            r_test = r_ant
            
        hum_cyl = crea_cilindro(preds_h_frame[a], preds_h_frame[b], r_test)
        
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()
        
        for rob_obj in robot_objs:
            # La libreria fcl.collide con la soglia di 130mm
            # Nota: se fcl non supporta la distanza minima nel collide, 
            # creiamo i cilindri del robot aumentati di 130mm per simulare la soglia del paper
            fcl.collide(hum_cyl, rob_obj, req, res)
            if res.is_collision:
                return True
                
    return False

def valuta_clip(i, targets_h_c, preds_h_c, targets_r_c, W_c, Cov_c, use_cache, cache_gt, mode="standard", m_base=0.0, tau=0.0):
    # GT e Baseline rimangono invariate per confronto
    gt_crash = cache_gt[i] if use_cache else check_collision_gt(targets_h_c, targets_r_c)
    
    pred_crash = False
    if mode == "standard":
        # Baseline del paper: cilindri predetti con raggio anatomico e soglia 13cm
        pred_crash = check_collision_pred_conformal(preds_h_c, targets_r_c, W_c, Cov_c)
    elif mode == "dynamic":
        # Nostro metodo con formule m_base e tau
        for h in range(25):
            p_prev = preds_h_c[h-1] if h > 0 else None
            if check_collision_dynamic_clipping(preds_h_c[h], p_prev, targets_r_c[h], W_c[h], Cov_c[h], m_base, tau):
                pred_crash = True
                break
    return (i, gt_crash, pred_crash)

def run_fcl_evaluation(results_file: str, config: dict = None, cache_file: str = 'gt_collisions_cache.pkl') -> None:
    """
    Fase 5: valuta le collisioni umano-robot tramite FCL e scrive il report finale.

    Modalità:
    - 'standard': usa check_collision_pred_conformal per ogni clip (ellissoidi fissi).
    - 'dynamic':  grid search su (m_base, tau) per trovare i parametri ottimali
                  del metodo con cilindri dinamicamente gonfiati.

    La cache GT (gt_collisions_cache.pkl o VAL_gt_collisions_cache.pkl) memorizza
    i risultati check_collision_gt per evitare di ricalcolarli a ogni run.

    Args:
        results_file: Percorso al pickle con 'targets_human', 'targets_robot',
                      'preds_human', 'predicted_thresholds_mah', 'covariances'.
        config:       Configurazione con 'collision_tuning' (mode, m_bases, taus).
        cache_file:   Percorso del file cache per le GT collisions.
    """
    exp_dir = os.path.dirname(results_file)
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
        
    targets_h = data['targets_human']
    targets_r = data['targets_robot']
    preds_h = data['preds_human']
    W = data['predicted_thresholds_mah']
    Cov = data['covariances']
    
    num_clips = targets_h.shape[0]
    
    use_cache = False
    cache_gt = None
    # Usiamo cache_file invece di FILE_CACHE_GT
    if os.path.exists(cache_file):
        use_cache = True
        with open(cache_file, 'rb') as f:
            cache_gt = pickle.load(f)

    log = get_logger()
    col_cfg = config.get('collision_tuning', {}) if config else {}
    mode = col_cfg.get('mode', 'standard')

    log.info(f"=== FASE 5: VALUTAZIONE COLLISIONI FCL ({mode.upper()}) ===")

    if mode == 'standard':
        log.info(f"Calcolo collisioni su {num_clips} clip (parallelo)...")
        risultati = Parallel(n_jobs=-1)(
                    delayed(valuta_clip)(i, targets_h[i], preds_h[i], targets_r[i], W[i], Cov[i], use_cache, cache_gt, mode="standard")
                    for i in tqdm(range(num_clips), desc="      [standard]")
        )
        risultati.sort(key=lambda x: x[0])
        gt_collisions_np = np.array([ris[1] for ris in risultati])
        best_pred_collisions_np = np.array([ris[2] for ris in risultati])
        
    elif mode == 'dynamic':
        m_bases = col_cfg.get('m_bases', [0.03])
        taus = col_cfg.get('taus', [0.1])
        miglior_f1 = 0.0
        migliori_params = None
        
        gt_collisions_np = None 
        best_pred_collisions_np = None

        log.info(f"Grid search su {len(m_bases) * len(taus)} combinazioni (m_bases × taus)...")
        for m_base in m_bases:
            for tau in taus:
                risultati = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(valuta_clip)(i, targets_h[i], preds_h[i], targets_r[i], W[i], Cov[i], use_cache, cache_gt, mode="dynamic", m_base=m_base, tau=tau) 
                    for i in range(num_clips)
                )
                risultati.sort(key=lambda x: x[0])
                
                if gt_collisions_np is None:
                    gt_collisions_np = np.array([ris[1] for ris in risultati])
                current_pred = np.array([ris[2] for ris in risultati])
                
                TP = np.sum(gt_collisions_np & current_pred)
                FP = np.sum((~gt_collisions_np) & current_pred)
                FN = np.sum(gt_collisions_np & (~current_pred))
                TN = np.sum((~gt_collisions_np) & (~current_pred))
                
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                log.info(f"[Grid] m_base={m_base}m, tau={tau}s -> TP:{TP} FP:{FP} FN:{FN} F1:{f1:.4f}")

                if f1 >= miglior_f1:
                    miglior_f1 = f1
                    migliori_params = (m_base, tau)
                    best_pred_collisions_np = current_pred.copy()

        log.info(f"Grid search completata. Vincitore: m_base={migliori_params[0]}m, tau={migliori_params[1]}s (F1: {miglior_f1:.4f})")
        
    # Salvataggio Cache GT e Calcolo finale sulle best predictions (comune a entrambi i mode)
    if not use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_collisions_np.tolist(), f)
        log.info(f"Cache GT salvata in '{cache_file}'.")
            
    TP = np.sum(gt_collisions_np & best_pred_collisions_np)
    FP = np.sum((~gt_collisions_np) & best_pred_collisions_np)
    FN = np.sum(gt_collisions_np & (~best_pred_collisions_np))
    TN = np.sum((~gt_collisions_np) & (~best_pred_collisions_np))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    log.info(f"--- RISULTATI FINALI ({mode.upper()}) ---")
    log.info(f"TP: {TP} | FP: {FP} | FN: {FN} | TN: {TN}")
    log.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    with open(os.path.join(exp_dir, 'fcl_evaluation_results.txt'), 'w') as f:
        f.write(f"Risultati Valutazione Collisioni FCL ({mode})\n")
        if mode == 'dynamic':
            f.write(f"Migliori Parametri: m_base={migliori_params[0]}m, tau={migliori_params[1]}s\n")
        f.write("-----------------------------------\n")
        f.write(f"TP: {TP}\nFP: {FP}\nFN: {FN}\nTN: {TN}\n")
        f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\n")