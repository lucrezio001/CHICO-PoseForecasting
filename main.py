import yaml
import os
import sys
import itertools
import time
import copy
import pickle
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Workaround: PyTorch 2.10+cu130 + PyTorch Lightning 2.6 hanno un bug di compatibilità
# su get_float32_matmul_precision() con API miste (legacy + new).
# Sopprimiamo il check di Lightning che lancia errore: non influenza le prestazioni.
try:
    import lightning_fabric.accelerators.cuda as _lf_cuda
    _lf_cuda._check_cuda_matmul_precision = lambda *args, **kwargs: None
except Exception:
    pass

from data_prep import process_offline_data
from score_extraction import process_and_save_scores
from train_regressor import train_phase3
from test_inference import run_test_inference
from evaluate_fcl import run_fcl_evaluation
from pipeline_utils import setup_logging, get_logger


def validate_config(config: dict) -> None:
    """
    Valida la configurazione prima di lanciare qualsiasi esperimento.
    Solleva SystemExit con messaggi chiari se ci sono errori, evitando
    crash silenziosi dopo ore di training.
    """
    errors = []

    # Campi obbligatori top-level
    for key in ('experiment_mode', 'active_model', 'directories', 'model_params'):
        if key not in config:
            errors.append(f"Chiave obbligatoria mancante: '{key}'")

    if errors:
        _fail_config(errors)

    mode = config['experiment_mode']
    model = config['active_model']

    if mode not in ('single', 'batch'):
        errors.append(f"experiment_mode deve essere 'single' o 'batch', trovato: '{mode}'")
    if model not in ('xgb', 'qrf', 'tabpfn', 'tabicl', 'lgbm', 'realmlp', 'realmlp_s'):
        errors.append(f"active_model deve essere 'xgb', 'qrf', 'tabpfn', 'tabicl', 'lgbm', 'realmlp' o 'realmlp_s', trovato: '{model}'")

    # Verifica esistenza file dataset
    dirs = config.get('directories', {})
    for key in ('val_data', 'test_data'):
        path = dirs.get(key, '')
        if not os.path.exists(path):
            errors.append(f"File dataset non trovato: directories.{key} = '{path}'")

    # Modalità batch: tutti i parametri del modello devono essere liste
    if mode == 'batch':
        batch_model_cfg = config.get('batch_run', {}).get(model, {})
        for param, val in batch_model_cfg.items():
            if not isinstance(val, list):
                errors.append(
                    f"batch_run.{model}.{param} deve essere una lista (anche con un solo valore), "
                    f"trovato: {type(val).__name__} = {val!r}"
                )
        batch_abl = config.get('batch_run', {}).get('ablation', {})
        for param, val in batch_abl.items():
            if not isinstance(val, list):
                errors.append(
                    f"batch_run.ablation.{param} deve essere una lista, "
                    f"trovato: {type(val).__name__} = {val!r}"
                )

    # Alpha deve essere in [0, 1]
    run_key = 'batch_run' if mode == 'batch' else 'single_run'
    run_model_cfg = config.get(run_key, {}).get(model, {})
    alphas = run_model_cfg.get('alpha', [])
    if not isinstance(alphas, list):
        alphas = [alphas]
    for a in alphas:
        if not (0.0 < a < 1.0):
            errors.append(f"alpha deve essere in (0, 1), trovato: {a}")

    _fail_config(errors)


def _fail_config(errors: list) -> None:
    if errors:
        print("\n❌ ERRORE CONFIGURAZIONE (config.yaml):")
        for err in errors:
            print(f"   - {err}")
        print()
        sys.exit(1)

def build_exp_name(config: dict) -> str:
    """
    Genera il nome univoco della directory per un esperimento.

    Il nome encode l'ablazione e i parametri del modello in forma compatta,
    così ogni combinazione di iperparametri corrisponde a una directory distinta.

    Formato ablation suffix: _knn{0|1}_dist{0|1}_time{0|1}_pw{N}_ts{H|C}
        knn  = use_knn_matrices
        dist = use_distances
        time = use_time_feature
        pw   = past_window
        ts   = temporal_strategy ('H'=per_horizon, 'C'=constant)
    """
    active_model = config['active_model']
    abl          = config['ablation']
    run_cfg      = config['current_run']
    alpha        = run_cfg.get('alpha', 0.1)

    abl_suffix = (
        f"_knn{'1' if abl['use_knn_matrices'] else '0'}"
        f"_dist{'1' if abl['use_distances'] else '0'}"
        f"_time{'1' if abl['use_time_feature'] else '0'}"
        f"_pw{abl['past_window']}"
        f"_ts{'H' if abl['temporal_strategy'] == 'per_horizon' else 'C'}"
    )

    if active_model == 'xgb':
        return (
            f"exp_xgb_a{alpha}"
            f"_est{run_cfg['n_estimators']}"
            f"_md{run_cfg['max_depth']}"
            f"_lr{run_cfg.get('learning_rate', 0.1)}"
            f"{abl_suffix}"
        )
    elif active_model == 'qrf':
        return (
            f"exp_qrf_a{alpha}"
            f"_est{run_cfg['n_estimators']}"
            f"_md{run_cfg['max_depth']}"
            f"{abl_suffix}"
        )
    elif active_model == 'lgbm':
        return (
            f"exp_lgbm_a{alpha}"
            f"_est{run_cfg.get('n_estimators', 100)}"
            f"_nl{run_cfg.get('num_leaves', 31)}"
            f"_lr{run_cfg.get('learning_rate', 0.1)}"
            f"{abl_suffix}"
        )
    elif active_model in ('realmlp', 'realmlp_s'):
        ep = run_cfg.get('n_epochs', 64)
        ep_part = f"_ep{ep}" if ep is not None else ""
        return f"exp_{active_model}_a{alpha}{ep_part}{abl_suffix}"
    elif active_model == 'tabicl':
        ss  = run_cfg.get('subsample_size', 0)
        pb  = run_cfg.get('predict_batch_size', 0)
        ss_part = f"_ss{ss}" if ss > 0 else ""
        pb_part = f"_pb{pb}" if pb > 0 else ""
        return f"exp_tabicl_a{alpha}{ss_part}{pb_part}{abl_suffix}"
    else:
        # tabpfn e altri con subsample/batch params
        ss  = run_cfg.get('subsample_size', 0)
        pb  = run_cfg.get('predict_batch_size', 0)
        ss_part = f"_ss{ss}" if ss > 0 else ""
        pb_part = f"_pb{pb}" if pb > 0 else ""
        return f"exp_{active_model}_a{alpha}{ss_part}{pb_part}{abl_suffix}"


def run_experiment(config: dict, offline_artifacts: dict, is_batch: bool = False) -> None:
    """
    Esegue un singolo esperimento completo dalla Fase 2.5 alla Fase 5.

    Fasi eseguite:
        2.5 — Calcolo non-conformity scores e covarianze (score_extraction)
        3   — Addestramento regressori quantile (train_regressor)
        3.5 — Valutazione FCL sul validation set
        4   — Inferenza batch sul test set (test_inference)
        5   — Valutazione collisioni FCL sul test set (evaluate_fcl)

    Il nome della directory è generato da build_exp_name() e codifica la
    combinazione di iperparametri, così esperimenti diversi non si sovrascrivono.

    Args:
        config:            Configurazione runtime con chiavi 'ablation', 'current_run',
                           'directories', 'active_model', 'model_params'.
        offline_artifacts: Output di process_offline_data (Fase 1–2).
        is_batch:          Se True, salva sotto 'batch_experiments/', altrimenti 'single_experiment/'.
    """
    t_start_total = time.time()

    base_dir     = config['directories']['results_base']
    active_model = config['active_model']
    flags        = config.get('pipeline_flags', {})
    skip_test    = flags.get('skip_test_inference', False)
    collision_only = flags.get('collision_only', False)

    exp_name = build_exp_name(config)
    exp_dir  = os.path.join(base_dir, "batch_experiments" if is_batch else "single_experiment", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log = setup_logging(exp_dir)
    log.info("=" * 50)
    log.info(f"ESPERIMENTO: {exp_name}")
    log.info("=" * 50)

    # --- MODALITA' COLLISION ONLY: salta tutto, riesegue solo Fase 5 ---
    if collision_only:
        bundle_path = flags.get('test_bundle_path', '').strip()
        if not bundle_path:
            bundle_path = os.path.join(exp_dir, 'test_results.pkl')
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                f"collision_only=true ma bundle non trovato: '{bundle_path}'\n"
                f"Imposta pipeline_flags.test_bundle_path nel config."
            )
        log.info(f"=== COLLISION ONLY: carico bundle da '{bundle_path}' ===")
        run_fcl_evaluation(bundle_path, config, cache_file='gt_collisions_cache.pkl')
        return

    # --- FASE 2.5: ESTRAZIONE SCORES ---
    nc_scores, cov_diagonals = process_and_save_scores(offline_artifacts, config, exp_dir)

    # --- FASE 3: ADDESTRAMENTO ---
    t_start_train = time.time()
    trained_models, val_results_bundle = train_phase3(config, exp_dir, offline_artifacts, nc_scores, cov_diagonals)
    t_train = time.time() - t_start_train

    # --- FASE 3.5: VALUTAZIONE FCL SUL VALIDATION SET ---
    log.info("=== FASE 3.5: VALUTAZIONE FCL SUL VALIDATION SET ===")
    val_bundle_path = os.path.join(exp_dir, 'val_results_bundle.pkl')
    with open(val_bundle_path, 'wb') as f:
        pickle.dump(val_results_bundle, f)

    run_fcl_evaluation(val_bundle_path, config, cache_file='VAL_gt_collisions_cache.pkl')

    # Libera risorse thread Windows dopo la grid search prima di avviare Fase 4.
    # Evita WinError 1450 (esaurimento nonpaged pool) su run lunghi.
    gc.collect()
    time.sleep(3)

    # Rinomina il file di risultati FCL del validation per non sovrascriverlo alla Fase 5
    old_txt = os.path.join(exp_dir, 'fcl_evaluation_results.txt')
    if os.path.exists(old_txt):
        os.replace(old_txt, os.path.join(exp_dir, 'VAL_fcl_evaluation_results.txt'))

    if skip_test:
        log.info("=== PIPELINE FLAGS: skip_test_inference=true → Fase 4 e 5 saltate ===")
        t_infer = 0.0
        num_test_clips = 0
        test_results_path = None
    else:
        # --- FASE 4: INFERENZA SUL TEST SET ---
        log.info("=== FASE 4: INFERENZA SUL TEST SET (PARALLELA) ===")
        t_start_infer = time.time()
        # Per TabPFN i modelli non vengono salvati su disco (in-context learning: troppo spazio,
        # troppo lento da serializzare). Li passiamo direttamente in-memory.
        models_in_memory = trained_models if active_model in ('tabpfn', 'tabicl') else None
        test_results_path, num_test_clips = run_test_inference(
            config, exp_dir, offline_artifacts, models_dict=models_in_memory
        )
        t_infer = time.time() - t_start_infer

        # --- FASE 5: VALUTAZIONE COLLISIONI FCL (TEST) ---
        log.info("=== FASE 5: VALUTAZIONE COLLISIONI FCL (TEST) ===")
        run_fcl_evaluation(test_results_path, config, cache_file='gt_collisions_cache.pkl')

    t_end_total = time.time()
    t_tot = t_end_total - t_start_total
    
    # Leggiamo le metriche da fcl_evaluation_results.txt per includerle nel report finale.
    # Il parsing è accoppiato al formato di scrittura in evaluate_fcl.py: se quel formato
    # cambia, aggiornare anche qui.
    fcl_res_path = os.path.join(exp_dir, 'fcl_evaluation_results.txt')
    metrics = {}
    if os.path.exists(fcl_res_path):
        with open(fcl_res_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.strip().split(':', 1)
                    metrics[key.strip()] = val.strip()
                    
    report_path = os.path.join(exp_dir, 'execution_performance.txt')
    with open(report_path, 'w') as f:
        f.write("=== REPORT PRESTAZIONI E COLLISIONI ===\n")
        f.write(f"Esperimento: {exp_name}\n\n")
        
        f.write("[METRICHE FCL COLLISION]\n")
        f.write(f"TP: {metrics.get('TP', '0')} | FP: {metrics.get('FP', '0')} | FN: {metrics.get('FN', '0')} | TN: {metrics.get('TN', '0')}\n")
        f.write(f"Precision: {metrics.get('Precision', '0.0000')}\n")
        f.write(f"Recall:    {metrics.get('Recall', '0.0000')}\n")
        f.write(f"F1 Score:  {metrics.get('F1 Score', '0.0000')}\n\n")
        
        f.write("[TEMPI DI ESECUZIONE]\n")
        f.write(f"Tempo Totale Esperimento:    {t_tot:.2f} secondi\n")
        f.write(f"Tempo Training (Fase 3):     {t_train:.2f} secondi\n")
        f.write(f"Tempo Inferenza Test(Fase 4):{t_infer:.2f} secondi (Su {num_test_clips} clip)\n")
        f.write("----------------------------------------\n")
        
        if num_test_clips > 0:
            avg_infer_clip = (t_infer / num_test_clips) * 1000
            avg_infer_frame = avg_infer_clip / 25.0
            f.write(f"MEDIA INFERENZA PER CLIP:    {avg_infer_clip:.2f} ms\n")
            f.write(f"MEDIA INFERENZA PER FRAME:   {avg_infer_frame:.2f} ms\n")

    log.info(f"Esperimento concluso in {t_tot/60:.2f} minuti.")
    log.info("Report finale salvato in 'execution_performance.txt'")

def main() -> None:
    """
    Entry point principale. Legge config.yaml e avvia gli esperimenti in
    modalità 'single' (test rapido) o 'batch' (grid search notturna).

    Modalità 'single':
        Esegue un singolo esperimento con i parametri di single_run.

    Modalità 'batch':
        Genera il prodotto cartesiano di ablation × parametri modello e
        lancia run_experiment per ogni combinazione.
        Nota: ogni parametro in batch_run deve essere una lista, anche se
        contiene un solo valore (es. alpha: [0.9]).
    """
    log = setup_logging()
    log.info("========================================= ")
    log.info("   COORDINATORE ESPERIMENTI CHICO 3D      ")
    log.info("=========================================")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    validate_config(config)

    offline_dir = os.path.join(config['directories']['results_base'], 'offline_workspace')
    offline_artifacts = process_offline_data(config['directories']['val_data'], offline_dir)

    mode         = config.get('experiment_mode', 'single')
    active_model = config.get('active_model', 'xgb')

    if mode == 'single':
        run_config = copy.deepcopy(config)
        run_config['ablation']    = config['single_run']['ablation']
        run_config['current_run'] = config['single_run'][active_model]
        run_experiment(run_config, offline_artifacts, is_batch=False)

    elif mode == 'batch':
        b = config['batch_run']

        abl_grid = b['ablation']
        abl_keys = list(abl_grid.keys())
        abl_vals = [abl_grid[k] for k in abl_keys]

        mod_grid = b[active_model]
        mod_keys = list(mod_grid.keys())
        mod_vals = [mod_grid[k] for k in mod_keys]

        all_keys = abl_keys + mod_keys
        all_vals = abl_vals + mod_vals
        combinazioni = list(itertools.product(*all_vals))

        log.info(f"Trovate {len(combinazioni)} combinazioni per il Batch Run ({active_model.upper()}).")

        for combo in combinazioni:
            combo_dict = dict(zip(all_keys, combo))

            run_config = copy.deepcopy(config)
            run_config['ablation'] = {k: combo_dict[k] for k in abl_keys}
            run_config['current_run'] = {k: combo_dict[k] for k in mod_keys}

            run_experiment(run_config, offline_artifacts, is_batch=True)

if __name__ == "__main__":
    main()