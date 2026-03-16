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
    Wrapper TabPFN per regressione in-context.

    TabPFN è un modello pre-addestrato che esegue "in-context learning":
    il fit() memorizza semplicemente i dati di training, e predict() li
    usa come contesto per la predizione bayesiana.

    LIMITAZIONE: TabPFNRegressor.predict() restituisce la media del posterior,
    non il quantile specificato da alpha. target_quantile è mantenuto per
    coerenza di interfaccia, ma non influenza il valore predetto.
    """
    def __init__(self, alpha: float, device: str = 'cpu'):
        self.target_quantile = 1.0 - alpha
        # Inizializziamo TabPFN. A differenza degli alberi, è già pre-addestrato 
        # e fa un "in-context learning" sui dati che gli passiamo nel fit!
        self.model = TabPFNRegressor(device=device)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # ATTENZIONE: TabPFN predice la media (o la mediana) del posterior,
        # NON il quantile self.target_quantile.
        # Il campo target_quantile è salvato per coerenza di interfaccia, ma
        # non viene usato internamente da TabPFNRegressor.predict().
        # Impatto: l'ellissoide conformal risulta più conservativo del necessario
        # per alpha < 0.5 (sovrastima dell'incertezza) e meno conservativo per alpha > 0.5.
        # Se la versione installata di TabPFN lo supporta, si può abilitare con:
        #   preds = self.model.predict(X, quantiles=[self.target_quantile])
        preds = self.model.predict(X)
        if preds.ndim > 1:
            preds = preds[:, 0]
        return preds

def build_regressor(config: dict):
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
            default_quantiles=[1.0 - alpha]
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
        # Creiamo il nuovo regressore TabPFN!
        return TabPFNQuantileWrapper(
            alpha=alpha,
            device=run_cfg.get('device', 'cpu')
        )
    else:
        raise ValueError(f"Regressore '{reg_type}' non riconosciuto.")