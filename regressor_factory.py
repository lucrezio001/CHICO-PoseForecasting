import numpy as np
import xgboost as xgb
from quantile_forest import RandomForestQuantileRegressor
# Importiamo TabPFN dal tuo pacchetto
from tabpfn import TabPFNRegressor 
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

class XGBQuantileWrapper:
    """
    Wrapper XGBoost per regressione quantile.

    Usa l'objective 'reg:quantileerror' di XGBoost per stimare il quantile
    (1 - alpha) della distribuzione target. L'output è il quantile superiore
    dell'intervallo di incertezza conformal.
    """
    def __init__(self, alpha: float, n_estimators: int, max_depth: int, random_state: int, xgb_kwargs: dict):
        self.target_quantile = 1.0 - alpha
        device = xgb_kwargs.get('device', 'cpu')
        
        self.model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=np.array([self.target_quantile]),
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            tree_method='hist',
            device=device,
            learning_rate=xgb_kwargs.get('learning_rate', 0.1),
            subsample=xgb_kwargs.get('subsample', 1.0),
            colsample_bytree=xgb_kwargs.get('colsample_bytree', 1.0),
            min_child_weight=xgb_kwargs.get('min_child_weight', 1),
            n_jobs=-1 if device == 'cpu' else None
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        if preds.ndim > 1:
            preds = preds[:, 0]
        return preds

class TabPFNQuantileWrapper:
    """
    Wrapper TabPFN per regressione quantile in-context.

    TabPFN è un modello pre-addestrato che esegue "in-context learning":
    il fit() memorizza semplicemente i dati di training, e predict() li
    usa come contesto per la predizione bayesiana.

    Usa output_type="quantiles" per ottenere il quantile (1 - alpha),
    identico al comportamento di XGBQuantileWrapper e QRF.

    Il parametro predict_batch_size divide il test set in batch durante
    l'inferenza per evitare CUDA OOM (il test set CHICO ha ~61k campioni,
    troppi per stare in una sola passata sulla GPU).
    """
    def __init__(self, alpha: float, device: str = 'cpu', predict_batch_size: int = 2000):
        self.target_quantile = 1.0 - alpha
        self.predict_batch_size = predict_batch_size
        # fit_mode="fit_with_cache": pre-calcola la KV representation del training set
        # durante fit() → predict() più veloce perché non ricalcola il contesto per ogni batch.
        self.model = TabPFNRegressor(device=device, fit_mode="fit_with_cache")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice in batch per evitare CUDA OOM su dataset di grandi dimensioni."""
        n = X.shape[0]
        if self.predict_batch_size <= 0 or n <= self.predict_batch_size:
            preds = self.model.predict(X, output_type="quantiles", quantiles=[self.target_quantile])
            return preds[0]

        results = []
        for start in range(0, n, self.predict_batch_size):
            batch = X[start : start + self.predict_batch_size]
            preds = self.model.predict(batch, output_type="quantiles", quantiles=[self.target_quantile])
            results.append(preds[0])
        return np.concatenate(results, axis=0)

def build_regressor(config: dict) -> XGBQuantileWrapper | TabPFNQuantileWrapper | object:
    """
    Factory: istanzia il regressore quantile specificato in config['active_model'].

    Modelli supportati:
    - 'qrf':    RandomForestQuantileRegressor (quantile_forest) — quantile nativo.
    - 'xgb':    XGBQuantileWrapper — XGBoost con objective 'reg:quantileerror'.
    - 'tabpfn': TabPFNQuantileWrapper — TabPFN pre-addestrato; NOTA: predice la
                media, non il quantile target (vedi docstring del wrapper).

    Il quantile effettivo addestrato è (1 - alpha), dove alpha è il livello di
    non-copertura della conformal prediction.

    Args:
        config: Configurazione runtime con 'active_model', 'current_run'
                (alpha, n_estimators, max_depth, ...) e 'model_params' (random_state).

    Returns:
        Istanza del regressore con interfaccia fit(X, y) / predict(X).
    """
    reg_type = config.get('active_model', 'xgb').lower()
    run_cfg = config['current_run']
    
    alpha = run_cfg['alpha']
    random_state = config['model_params']['random_state']

    if reg_type == 'qrf':
        return RandomForestQuantileRegressor(
            n_estimators=run_cfg['n_estimators'],
            max_depth=run_cfg['max_depth'],
            random_state=random_state,
            default_quantiles=[1.0 - alpha],
            n_jobs=-1,
        )
    elif reg_type == 'xgb':
        return XGBQuantileWrapper(
            alpha=alpha,
            n_estimators=run_cfg['n_estimators'],
            max_depth=run_cfg['max_depth'],
            random_state=random_state,
            xgb_kwargs=run_cfg 
        )
    elif reg_type == 'tabpfn':
        return TabPFNQuantileWrapper(
            alpha=alpha,
            device=run_cfg.get('device', 'cpu'),
            predict_batch_size=run_cfg.get('predict_batch_size', 2000),
        )
    else:
        raise ValueError(f"Regressore '{reg_type}' non riconosciuto.")