import os
import pickle
import numpy as np
import fcl
from tqdm import tqdm
from joblib import Parallel, delayed
from pipeline_utils import get_logger, FRAME_RATE_S

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
CONN_ROBOT = [(i, i + 1) for i in range(8)]
# Raggio approssimato dei link del robot collaborativo (mm).
# Valore conservativo empirico per cobot industriali a braccio circolare (~4 cm).
RAGGIO_ROBOT = 40.0  # mm


def get_human_radius(joint_a: int, joint_b: int) -> float:
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


def get_joint_radius(joint_idx: int) -> float:
    """Calcola il raggio anatomico massimo tra i segmenti collegati a un giunto."""
    raggi = []
    for (a, b) in CONN_HUMAN:
        if joint_idx == a or joint_idx == b:
            raggi.append(get_human_radius(a, b))
    if not raggi:
        return 50.0
    return max(raggi)


# Lookup precomputato: JOINT_RADII[j] = raggio anatomico massimo per il giunto j (mm).
JOINT_RADII = {j: get_joint_radius(j) for j in range(15)}

# Lookup precomputato: SEGMENT_RADII[(a,b)] = raggio anatomico del segmento osseo (mm).
# Evita di chiamare get_human_radius() nei loop interni.
SEGMENT_RADII = {(a, b): get_human_radius(a, b) for a, b in CONN_HUMAN}
_MAX_ANAT_RADIUS = max(SEGMENT_RADII.values())

# Array numpy per calcolo vettorizzato dei centri dei link robot.
_CONN_ROBOT_A = np.array([a for a, b in CONN_ROBOT])
_CONN_ROBOT_B = np.array([b for a, b in CONN_ROBOT])

# Array numpy per calcolo vettorizzato dei segmenti umani (usati in precompute_dynamic_tensors).
_SEG_A = np.array([a for a, b in CONN_HUMAN])  # (16,) — indice giunto A per ogni segmento
_SEG_B = np.array([b for a, b in CONN_HUMAN])  # (16,) — indice giunto B per ogni segmento


def crea_cilindro(p1: np.ndarray, p2: np.ndarray, raggio: float) -> fcl.CollisionObject:
    """Crea un cilindro fisico orientato tra p1 e p2 con il raggio specificato (mm)."""
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


def crea_ellissoide_statistico(centro: np.ndarray, cov_matrix: np.ndarray, s_hat: float) -> fcl.CollisionObject:
    """
    Crea un ellissoide FCL orientato secondo la decomposizione SVD della covarianza.

    I semi-assi sono: sqrt(s_hat * lambda_i) dove lambda_i sono gli autovalori di cov_matrix.
    Rappresenta la regione di incertezza conformal al livello s_hat.
    """
    U, S, Vh = np.linalg.svd(cov_matrix)

    # Semi-assi: radice(soglia * autovalore). Aggiungiamo 1e-3 per stabilità di FCL
    a = np.sqrt(s_hat * S[0]) + 1e-3
    b = np.sqrt(s_hat * S[1]) + 1e-3
    c = np.sqrt(s_hat * S[2]) + 1e-3

    # Correzione del determinante (FCL esplode se non è una rotazione pura)
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1

    return fcl.CollisionObject(fcl.Ellipsoid(a, b, c), fcl.Transform(U, centro))


def check_collision_gt(pose_umano: np.ndarray, pose_robot: np.ndarray) -> bool:
    """
    GROUND TRUTH: verifica collisione tramite segmenti ossei (cilindri) vs link robot (cilindri).

    Collisione = d(cilindro_umano, cilindro_robot) < SOGLIA_COLLISIONE in qualsiasi frame.

    Args:
        pose_umano: Shape (T, 15, 3) — posizioni giunti umani per T frame.
        pose_robot: Shape (T, 9, 3)  — posizioni giunti robot per T frame.

    Returns:
        True se almeno un frame contiene una collisione.
    """
    num_frames = pose_umano.shape[0]
    req, res = fcl.DistanceRequest(), fcl.DistanceResult()
    for t in range(num_frames):
        robot_objs = [crea_cilindro(pose_robot[t, a], pose_robot[t, b], RAGGIO_ROBOT) for a, b in CONN_ROBOT]
        for (a, b) in CONN_HUMAN:
            human_obj = crea_cilindro(pose_umano[t, a], pose_umano[t, b], SEGMENT_RADII[(a, b)])
            for r_obj in robot_objs:
                if fcl.distance(human_obj, r_obj, req, res) <= SOGLIA_COLLISIONE:
                    return True
    return False


def check_collision_pred_conformal(
    pose_umano_pred: np.ndarray,
    pose_robot: np.ndarray,
    W_clip: np.ndarray,
    Cov_clip: np.ndarray,
) -> bool:
    """
    PREDIZIONE STANDARD: giunti come ellissoidi statistici vs link robot come cilindri.

    Collisione = d(ellissoide_j, cilindro_robot) < 130mm + r_anat,j in qualsiasi frame.

    Args:
        pose_umano_pred: Shape (T, 15, 3) — predizioni giunti umani.
        pose_robot:      Shape (T, 9, 3)  — posizioni robot ground truth.
        W_clip:          Shape (T, 15)    — soglie Mahalanobis predette (Q_hat).
        Cov_clip:        Shape (T, 15, 3, 3) — matrici di covarianza per giunto.

    Returns:
        True se almeno un frame/giunto genera un allarme di prossimità.
    """
    num_frames = pose_umano_pred.shape[0]
    req, res = fcl.DistanceRequest(), fcl.DistanceResult()
    for t in range(num_frames):
        robot_objs = [crea_cilindro(pose_robot[t, a], pose_robot[t, b], RAGGIO_ROBOT) for a, b in CONN_ROBOT]
        for j in range(15):
            s_hat = max(W_clip[t, j], 0.0)
            human_obj = crea_ellissoide_statistico(pose_umano_pred[t, j], Cov_clip[t, j], s_hat)
            soglia_dinamica = SOGLIA_COLLISIONE + JOINT_RADII[j]
            for r_obj in robot_objs:
                if fcl.distance(human_obj, r_obj, req, res) <= soglia_dinamica:
                    return True
    return False


def _build_robot_objects(targets_r_frame: np.ndarray) -> list:
    """Costruisce oggetti FCL per i link robot di un singolo frame. Shape input: (9, 3)."""
    return [crea_cilindro(targets_r_frame[a], targets_r_frame[b], RAGGIO_ROBOT) for a, b in CONN_ROBOT]


def _frame_quick_reject(preds_h_frame: np.ndarray, targets_r_frame: np.ndarray, max_margin: float) -> bool:
    """
    Pre-filtro AABB conservativo. Ritorna True se nessuna collisione è geometricamente
    possibile nel frame, permettendo di saltare tutti i check FCL.

    Il margine max_margin deve essere un bound superiore garantito su qualsiasi raggio
    di rilevamento, quindi il filtro non può mai produrre falsi negativi (collisioni mancate).
    """
    h_min = preds_h_frame.min(axis=0) - max_margin
    h_max = preds_h_frame.max(axis=0) + max_margin
    r_min = targets_r_frame.min(axis=0)
    r_max = targets_r_frame.max(axis=0)
    return bool(np.any(h_min > r_max) or np.any(h_max < r_min))


def check_collision_conformal_frame(pose_h_frame, W_frame, Cov_frame, robot_objs):
    """
    Stage 1 — Meccanismo di attenzione (singolo frame).

    Verifica se almeno un ellissoide conformal del giunto j si avvicina a un link
    robot entro la soglia: d(ellissoide_j, cyl_robot) < 130mm + r_anat,j.
    Usa robot_objs precostruiti per non riallocare oggetti FCL.
    """
    req, res = fcl.DistanceRequest(), fcl.DistanceResult()
    for j in range(15):
        s_hat = max(W_frame[j], 0.0)
        human_obj = crea_ellissoide_statistico(pose_h_frame[j], Cov_frame[j], s_hat)
        soglia_j = SOGLIA_COLLISIONE + JOINT_RADII[j]
        for r_obj in robot_objs:
            if fcl.distance(human_obj, r_obj, req, res) <= soglia_j:
                return True
    return False


def check_collision_dynamic_frame(
    preds_h_frame, preds_h_prev,
    targets_r_frame, targets_r_prev,
    W_frame, Cov_frame,
    robot_objs, allarme_conformal,
    m_base_mm, tau
):
    """
    Stage 2 — Collisione con cilindro gonfiato via soft-clipping (singolo frame).

    Formula (tutte le grandezze in mm e mm/s):

        R_cp,j   = sqrt(W[j] * lambda_max(Cov[j]))       # semiasse maggiore ellissoide CP
        V_rel    = ||V_umano_seg - V_robot_link||         # velocità relativa 3D specifica
        m_t      = m_base + tau * V_rel                   # soglia fisica dinamica
        R_soft   = m_t * tanh(R_cp / m_t)                # soft-clipping Catoni-Giulini
        R_umano  = r_anat + R_soft                        # raggio finale capsula FCL

    Collisione: fcl.distance(capsula_umana, cyl_robot) <= 130mm.
    Se allarme_conformal=False, si usa solo r_anat (baseline senza gonfiamento).

    Proprietà garantita: R_soft <= R_cp (tanh(x) <= x per x >= 0),
    quindi il margine del pre-filtro AABB è sempre conservativo.
    """
    # Precompute R_cp per giunto: sqrt(W[j] * lambda_max(Cov[j]))
    # lambda_max è il primo valore singolare (SVD di matrice PSD = autovalori)
    R_cp = np.array([
        np.sqrt(max(W_frame[j], 0.0) * np.linalg.svd(Cov_frame[j], compute_uv=False)[0])
        for j in range(15)
    ])  # mm

    # Pre-filtro AABB: skip sicuro se i bounding box sono chiaramente separati.
    # max_margin = distanza massima di rilevamento possibile per qualsiasi coppia.
    # Poiché R_soft <= R_cp (proprietà tanh), usiamo R_cp.max() come bound superiore.
    max_margin = SOGLIA_COLLISIONE + _MAX_ANAT_RADIUS + RAGGIO_ROBOT + float(R_cp.max())
    if _frame_quick_reject(preds_h_frame, targets_r_frame, max_margin):
        return False

    # Velocità umane per giunto: vettori 3D (mm/s)
    if preds_h_prev is not None:
        vel_human = (preds_h_frame - preds_h_prev) / FRAME_RATE_S  # (15, 3)
    else:
        vel_human = np.zeros((15, 3))

    # Velocità centri link robot: (8, 3) mm/s
    # Usa _CONN_ROBOT_A/B per calcolo vettorizzato senza loop Python
    if targets_r_prev is not None:
        r_curr = 0.5 * (targets_r_frame[_CONN_ROBOT_A] + targets_r_frame[_CONN_ROBOT_B])
        r_prev = 0.5 * (targets_r_prev[_CONN_ROBOT_A] + targets_r_prev[_CONN_ROBOT_B])
        vel_robot = (r_curr - r_prev) / FRAME_RATE_S  # (8, 3)
    else:
        vel_robot = np.zeros((8, 3))

    req = fcl.DistanceRequest()

    for a, b in CONN_HUMAN:
        r_anat = SEGMENT_RADII[(a, b)]

        if allarme_conformal:
            # R_cp del segmento: massimo tra i due giunti estremi
            R_cp_seg = max(R_cp[a], R_cp[b])
            # Velocità del centro del segmento umano (vettore 3D)
            v_h = 0.5 * (vel_human[a] + vel_human[b])

        for rob_idx, rob_obj in enumerate(robot_objs):
            if allarme_conformal:
                # V_rel: velocità relativa tra segmento umano e link robot specifico
                V_rel = np.linalg.norm(v_h - vel_robot[rob_idx])
                m_t = m_base_mm + tau * V_rel
                # Soft-clipping asintotico: R_soft -> R_cp per Rcp << m_t (lineare),
                # R_soft -> m_t per Rcp >> m_t (saturazione). Sempre R_soft <= R_cp.
                R_soft = m_t * np.tanh(R_cp_seg / m_t) if m_t > 1e-9 else 0.0
                r_test = r_anat + R_soft
            else:
                r_test = r_anat

            hum_cyl = crea_cilindro(preds_h_frame[a], preds_h_frame[b], r_test)
            res = fcl.DistanceResult()
            if fcl.distance(hum_cyl, rob_obj, req, res) <= SOGLIA_COLLISIONE:
                return True

    return False


def precompute_dynamic_tensors(
    preds_h_c: np.ndarray,
    targets_r_c: np.ndarray,
    W_c: np.ndarray,
    Cov_c: np.ndarray,
) -> dict:
    """
    Pre-calcola in forma vettorizzata le quantità usate dal metodo dinamico per tutti
    i 25 frame contemporaneamente, sfruttando la disponibilità simultanea di:
      - predizioni rete: preds_h_c (25, 15, 3)
      - robot GT:        targets_r_c (25, 9, 3) — noto, zero data leakage

    Ritorna un dict con:
        'vel_h':    (25, 15, 3)  — velocità per giunto umano (da predizioni)
        'vel_r':    (25, 8, 3)   — velocità per link robot (da GT)
        'R_cp':     (25, 15)     — raggio conformal per giunto
        'R_cp_seg': (25, 16)     — raggio conformal per segmento (max tra giunti estremi)
        'V_rel':    (25, 16, 8)  — velocità relativa segmento×link
    """
    T = preds_h_c.shape[0]  # 25

    # --- Velocità umane (da predizioni rete) ---
    # Frame 0: nessun frame precedente → velocità zero
    vel_h = np.zeros((T, 15, 3))
    vel_h[1:] = np.diff(preds_h_c, axis=0) / FRAME_RATE_S  # (T-1, 15, 3)

    # --- Velocità robot (da GT — nota, nessun data leakage) ---
    r_centers = 0.5 * (targets_r_c[:, _CONN_ROBOT_A] + targets_r_c[:, _CONN_ROBOT_B])  # (T, 8, 3)
    vel_r = np.zeros((T, 8, 3))
    vel_r[1:] = np.diff(r_centers, axis=0) / FRAME_RATE_S  # (T-1, 8, 3)

    # --- R_cp per giunto: sqrt(W[h,j] * lambda_max(Cov[h,j])) — SVD vettorizzata ---
    sv_all = np.linalg.svd(Cov_c, compute_uv=False)   # (T, 15, 3)
    lambda_max_all = sv_all[..., 0]                     # (T, 15)
    R_cp = np.sqrt(np.maximum(W_c, 0.0) * lambda_max_all)  # (T, 15)

    # --- R_cp per segmento: max tra i due giunti estremi ---
    R_cp_seg = np.maximum(R_cp[:, _SEG_A], R_cp[:, _SEG_B])  # (T, 16)

    # --- Velocità per segmento umano: media tra giunti A e B ---
    v_seg = 0.5 * (vel_h[:, _SEG_A] + vel_h[:, _SEG_B])  # (T, 16, 3)

    # --- V_rel per (frame, segmento, link_robot): broadcasting ---
    # v_seg:  (T, 16, 3) → (T, 16, 1, 3)
    # vel_r:  (T, 8, 3)  → (T, 1, 8, 3)
    v_diff = v_seg[:, :, np.newaxis, :] - vel_r[:, np.newaxis, :, :]  # (T, 16, 8, 3)
    V_rel = np.linalg.norm(v_diff, axis=-1)  # (T, 16, 8)

    return {
        'vel_h':    vel_h,
        'vel_r':    vel_r,
        'R_cp':     R_cp,
        'R_cp_seg': R_cp_seg,
        'V_rel':    V_rel,
    }


def check_collision_dynamic_frame_v2(
    preds_h_frame: np.ndarray,
    targets_r_frame: np.ndarray,
    robot_objs: list,
    allarme_conformal: bool,
    m_base_mm: float,
    tau: float,
    R_cp_h: np.ndarray,
    R_cp_seg_h: np.ndarray,
    V_rel_h: np.ndarray,
) -> bool:
    """
    Stage 2 — Collisione con cilindro gonfiato via soft-clipping (singolo frame),
    versione ottimizzata con quantità pre-calcolate.

    Rispetto a check_collision_dynamic_frame():
      - Non ricalcola velocità o R_cp (già forniti da precompute_dynamic_tensors)
      - Esegue solo le operazioni scalari/FCL strettamente necessarie

    Args:
        preds_h_frame:   (15, 3) — predizioni giunti umani al frame h
        targets_r_frame: (9, 3)  — posizioni robot GT al frame h
        robot_objs:      list    — oggetti FCL robot (pre-costruiti)
        allarme_conformal: bool  — True se Stage 1 ha rilevato prossimità
        m_base_mm:       float   — soglia base in mm
        tau:             float   — guadagno velocità
        R_cp_h:          (15,)   — R_cp per giunto al frame h
        R_cp_seg_h:      (16,)   — R_cp per segmento al frame h
        V_rel_h:         (16, 8) — velocità relativa seg×link al frame h
    """
    # Pre-filtro AABB: bound superiore usando R_cp massimo (R_soft <= R_cp sempre)
    max_margin = SOGLIA_COLLISIONE + _MAX_ANAT_RADIUS + RAGGIO_ROBOT + float(R_cp_h.max())
    if _frame_quick_reject(preds_h_frame, targets_r_frame, max_margin):
        return False

    req = fcl.DistanceRequest()

    for seg_idx, (a, b) in enumerate(CONN_HUMAN):
        r_anat = SEGMENT_RADII[(a, b)]

        if allarme_conformal:
            R_cp_seg = R_cp_seg_h[seg_idx]

        for rob_idx, rob_obj in enumerate(robot_objs):
            if allarme_conformal:
                V_rel = V_rel_h[seg_idx, rob_idx]
                m_t = m_base_mm + tau * V_rel
                R_soft = m_t * np.tanh(R_cp_seg / m_t) if m_t > 1e-9 else 0.0
                r_test = r_anat + R_soft
            else:
                r_test = r_anat

            hum_cyl = crea_cilindro(preds_h_frame[a], preds_h_frame[b], r_test)
            res = fcl.DistanceResult()
            if fcl.distance(hum_cyl, rob_obj, req, res) <= SOGLIA_COLLISIONE:
                return True

    return False


def valuta_clip(
    i: int,
    targets_h_c: np.ndarray,
    preds_h_c: np.ndarray,
    targets_r_c: np.ndarray,
    W_c: np.ndarray,
    Cov_c: np.ndarray,
    use_cache: bool,
    cache_gt: list | None,
    mode: str = "standard",
    m_base: float = 0.0,
    tau: float = 0.0,
) -> tuple[int, bool, bool]:
    gt_crash = cache_gt[i] if use_cache else check_collision_gt(targets_h_c, targets_r_c)

    pred_crash = False
    if mode == "standard":
        pred_crash = check_collision_pred_conformal(preds_h_c, targets_r_c, W_c, Cov_c)
    elif mode == "dynamic":
        m_base_mm = m_base * 1000.0
        # Pre-calcolo vettorizzato di velocità, R_cp e V_rel per tutti i 25 frame
        precomp = precompute_dynamic_tensors(preds_h_c, targets_r_c, W_c, Cov_c)
        for h in range(25):
            # Robot objects costruiti UNA VOLTA per frame, riusati in Stage 1 e Stage 2
            robot_objs = _build_robot_objects(targets_r_c[h])
            allarme = check_collision_conformal_frame(preds_h_c[h], W_c[h], Cov_c[h], robot_objs)
            if check_collision_dynamic_frame_v2(
                preds_h_c[h], targets_r_c[h],
                robot_objs, allarme,
                m_base_mm, tau,
                R_cp_h=precomp['R_cp'][h],
                R_cp_seg_h=precomp['R_cp_seg'][h],
                V_rel_h=precomp['V_rel'][h],
            ):
                pred_crash = True
                break

    return (i, gt_crash, pred_crash)


def run_fcl_evaluation(results_file: str, config: dict = None, cache_file: str = 'gt_collisions_cache.pkl') -> None:
    """
    Fase 5: valuta le collisioni umano-robot tramite FCL e scrive il report finale.

    Modalità:
    - 'standard': usa check_collision_pred_conformal per ogni clip (ellissoidi fissi).
    - 'dynamic':  grid search su (m_base, tau) per trovare i parametri ottimali
                  del metodo con cilindri dinamicamente gonfiati via soft-clipping.

    La cache GT memorizza i risultati check_collision_gt per evitare ricalcoli.

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
        grid_pairs = [(m, t) for m in m_bases for t in taus]
        for m_base, tau in tqdm(grid_pairs, desc="      [dynamic grid]", unit="config"):
            risultati = list(tqdm(
                Parallel(n_jobs=-1, prefer="threads", return_as="generator")(
                    delayed(valuta_clip)(i, targets_h[i], preds_h[i], targets_r[i], W[i], Cov[i], use_cache, cache_gt, mode="dynamic", m_base=m_base, tau=tau)
                    for i in range(num_clips)
                ),
                total=num_clips,
                desc=f"        m={m_base} τ={tau}",
                unit="clip",
                leave=False,
            ))
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

    # Salvataggio Cache GT (comune a entrambi i mode)
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
