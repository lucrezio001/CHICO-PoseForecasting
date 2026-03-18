import numpy as np
import xgboost as xgb
from quantile_forest import RandomForestQuantileRegressor
from tabpfn import TabPFNRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="pytabkit")

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

class TabICLQuantileWrapper:
    """
    Wrapper TabICLv2 (tabicl) per regressione quantile zero-shot.

    TabICLv2 è un foundation model pre-addestrato (checkpoint Hugging Face):
    - fit() è solo preprocessing dei dati (istantaneo, scarica checkpoint 1 sola volta)
    - predict() esegue in-context learning transformer → nessun training necessario
    - 10x più veloce di TabPFN-2.5 su grandi test set grazie a Flash Attention + AMP

    Differenza da TabPFN:
    - TabPFN: fit memorizza dati, predict fa attention O(N_train × N_batch)
    - TabICL: fit prepara preprocessing, predict usa architettura più efficiente
    - Supporta `output_type="quantiles"` per predizione nativa di quantili multipli

    Limite: performance zero-shot migliore con N_train > 300 campioni.
    """

    def __init__(self, alpha: float, device: str | None = None, predict_batch_size: int = 4000):
        self.target_quantile = round(1.0 - alpha, 4)
        self.predict_batch_size = predict_batch_size
        from tabicl import TabICLRegressor
        self.model = TabICLRegressor(device=device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabICLQuantileWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice il quantile (1-alpha) in batch per evitare OOM.

        Returns:
            np.ndarray shape (n_samples,)
        """
        n = X.shape[0]
        if self.predict_batch_size <= 0 or n <= self.predict_batch_size:
            out = self.model.predict(X, output_type="quantiles", alphas=[self.target_quantile])
            return out[:, 0].ravel()

        results = []
        for start in range(0, n, self.predict_batch_size):
            batch = X[start : start + self.predict_batch_size]
            out = self.model.predict(batch, output_type="quantiles", alphas=[self.target_quantile])
            results.append(out[:, 0])
        return np.concatenate(results, axis=0)


class LGBMQuantileWrapper:
    """
    Wrapper LightGBM per regressione quantile.

    Usa l'objective 'quantile' di LightGBM per stimare il quantile (1-alpha).
    Vantaggi rispetto a XGB: spesso più veloce su CPU, ottimo per dataset medi.
    Non richiede GPU ma scala bene su tutti i core CPU disponibili.
    """

    def __init__(
        self,
        alpha: float,
        n_estimators: int = 100,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        import lightgbm as lgb
        self.target_quantile = 1.0 - alpha
        self.model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=self.target_quantile,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMQuantileWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns:
            np.ndarray shape (n_samples,) — quantile (1-alpha) per ogni campione.
        """
        return self.model.predict(X).ravel()


class RealMLPQuantileWrapper:
    """
    Wrapper RealMLP (pytabkit) per regressione quantile tramite pinball loss.

    Supporta due varianti:
    - 'td'  : RealMLP_TD_Regressor  — rete completa, più espressiva, ~7-8 ore/375 modelli
    - 'td_s': RealMLP_TD_S_Regressor — rete piccola, più veloce,    ~1-2 ore/375 modelli

    Il parametro n_epochs limita il numero massimo di epoche (default libreria: 256).
    Ridurre a 50-100 velocizza significativamente senza perdita grave di accuratezza.
    """

    def __init__(
        self,
        alpha: float,
        device: str = "cpu",
        random_state: int = 42,
        variant: str = "td_s",
        n_epochs: int | None = 64,
    ):
        self.target_quantile = 1.0 - alpha
        kwargs = dict(
            train_metric_name=f"pinball({self.target_quantile:.4f})",
            device=device,
            random_state=random_state,
        )
        if n_epochs is not None:
            kwargs["n_epochs"] = n_epochs

        if variant == "td_s":
            from pytabkit import RealMLP_TD_S_Regressor
            self.model = RealMLP_TD_S_Regressor(**kwargs)
        else:
            from pytabkit import RealMLP_TD_Regressor
            self.model = RealMLP_TD_Regressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RealMLPQuantileWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns:
            np.ndarray shape (n_samples,) — quantile (1-alpha) per ogni campione.
        """
        return self.model.predict(X).ravel()


def build_regressor(config: dict) -> XGBQuantileWrapper | TabPFNQuantileWrapper | TabICLQuantileWrapper | LGBMQuantileWrapper | RealMLPQuantileWrapper | object:
    """
    Factory: istanzia il regressore quantile specificato in config['active_model'].

    Modelli supportati:
    - 'qrf':      RandomForestQuantileRegressor (quantile_forest) — quantile nativo.
    - 'xgb':      XGBQuantileWrapper — XGBoost con objective 'reg:quantileerror'.
    - 'tabpfn':   TabPFNQuantileWrapper — TabPFN in-context learning (KV cache).
    - 'tabicl':   TabICLQuantileWrapper — TabICLv2 zero-shot foundation model (10x più veloce di TabPFN).
    - 'lgbm':     LGBMQuantileWrapper — LightGBM con objective 'quantile'. CPU-only, veloce.
    - 'realmlp':  RealMLPQuantileWrapper — RealMLP_TD (pytabkit) con pinball loss.
    - 'realmlp_s':RealMLPQuantileWrapper — RealMLP_TD_S (small, più veloce) con pinball loss.

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
    elif reg_type == 'tabicl':
        return TabICLQuantileWrapper(
            alpha=alpha,
            device=run_cfg.get('device', None),  # None = auto-detect GPU
            predict_batch_size=run_cfg.get('predict_batch_size', 4000),
        )
    elif reg_type == 'lgbm':
        return LGBMQuantileWrapper(
            alpha=alpha,
            n_estimators=run_cfg.get('n_estimators', 100),
            num_leaves=run_cfg.get('num_leaves', 31),
            max_depth=run_cfg.get('max_depth', -1),
            learning_rate=run_cfg.get('learning_rate', 0.1),
            random_state=random_state,
        )
    elif reg_type in ('realmlp', 'realmlp_s'):
        return RealMLPQuantileWrapper(
            alpha=alpha,
            device=run_cfg.get('device', 'cpu'),
            random_state=random_state,
            variant='td_s' if reg_type == 'realmlp_s' else 'td',
            n_epochs=run_cfg.get('n_epochs', 64),
        )
    else:
        raise ValueError(f"Regressore '{reg_type}' non riconosciuto. Scegli tra: qrf, xgb, tabpfn, tabicl, lgbm, realmlp, realmlp_s.")