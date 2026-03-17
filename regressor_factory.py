import numpy as np
import xgboost as xgb
from quantile_forest import RandomForestQuantileRegressor
# Importiamo TabPFN dal tuo pacchetto
from tabpfn import TabPFNRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="pgbm")

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

class PGBMQuantileWrapper:
    """
    Wrapper PGBM (Probabilistic Gradient Boosting Machines) per regressione quantile.

    PGBM fitta un singolo modello che, oltre alla predizione puntuale, stima la
    varianza foglia-per-foglia e assume una distribuzione parametrica sull'output
    (es. 'normal', 'lognormal', 'gamma'). I quantili si estraggono campionando
    dalla distribuzione stimata con n_estimates campioni Monte Carlo.

    Differenza chiave rispetto a XGB/QRF:
    - XGB/QRF: minimizzano la pinball loss direttamente → precisi ma un modello per quantile
    - PGBM: un solo fit, poi quantili via sampling → più veloce in training, dipende
            da quanto bene la distribuzione scelta approssima i residui reali.

    Per i NC scores Mahalanobis (valori positivi, asimmetrici a destra) la distribuzione
    'lognormal' o 'gamma' è generalmente migliore di 'normal'.

    Backend:
    - device='cpu'  → pgbm.sklearn.HistGradientBoostingRegressor (sklearn-compatible)
    - device='cuda' → pgbm.torch.PGBMRegressor (GPU, richiede CUDA >= 10.2)
    """

    def __init__(
        self,
        alpha: float,
        n_estimators: int = 500,
        distribution: str = "lognormal",
        n_estimates: int = 1000,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.target_quantile = 1.0 - alpha
        self.n_estimates      = n_estimates
        self.random_state     = random_state
        self.device           = device.lower()

        if self.device in ("cuda", "gpu"):
            # Torch backend: supporta GPU
            from pgbm.torch import PGBMRegressor
            self.backend = "torch"
            self.model = PGBMRegressor(
                n_estimators=n_estimators,
                distribution=distribution,
                device="gpu",
                seed=random_state,
                verbose=0,
            )
        else:
            # Sklearn backend: CPU, sklearn-compatible
            from pgbm.sklearn import HistGradientBoostingRegressor as PGBMSklearn
            self.backend = "sklearn"
            self.model = PGBMSklearn(
                n_iter=n_estimators,
                distribution=distribution,
                with_variance=True,
                random_state=random_state,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PGBMQuantileWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice il quantile target via campionamento Monte Carlo dalla distribuzione stimata.

        Returns:
            np.ndarray shape (n_samples,) — quantile (1-alpha) per ogni campione.
        """
        if self.backend == "torch":
            yhat_dist = self.model.predict_dist(X, n_forecasts=self.n_estimates)
            # yhat_dist shape: (n_estimates, n_samples) — potrebbe essere torch.Tensor
            if hasattr(yhat_dist, "cpu"):
                yhat_dist = yhat_dist.cpu().numpy()
        else:
            yhat, yhat_std = self.model.predict(X, return_std=True)
            yhat_dist = self.model.sample(
                yhat, yhat_std,
                n_estimates=self.n_estimates,
                random_state=self.random_state,
            )
            # yhat_dist shape: (n_estimates, n_samples)

        return np.percentile(yhat_dist, self.target_quantile * 100.0, axis=0)


def build_regressor(config: dict) -> XGBQuantileWrapper | TabPFNQuantileWrapper | PGBMQuantileWrapper | object:
    """
    Factory: istanzia il regressore quantile specificato in config['active_model'].

    Modelli supportati:
    - 'qrf':    RandomForestQuantileRegressor (quantile_forest) — quantile nativo.
    - 'xgb':    XGBQuantileWrapper — XGBoost con objective 'reg:quantileerror'.
    - 'tabpfn': TabPFNQuantileWrapper — TabPFN pre-addestrato in-context learning.
    - 'pgbm':   PGBMQuantileWrapper — PGBM distribuzionale, quantile via sampling MC.

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
    elif reg_type == 'pgbm':
        return PGBMQuantileWrapper(
            alpha=alpha,
            n_estimators=run_cfg.get('n_estimators', 500),
            distribution=run_cfg.get('distribution', 'lognormal'),
            n_estimates=run_cfg.get('n_estimates', 1000),
            device=run_cfg.get('device', 'cpu'),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Regressore '{reg_type}' non riconosciuto. Scegli tra: qrf, xgb, tabpfn, pgbm.")