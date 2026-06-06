import importlib
from omegaconf import DictConfig


def train_model(X_train, y_train, model_cfg: DictConfig):
    """
    Dynamically loads and trains the model class configured in model_cfg.
    """
    class_name = model_cfg.class_name
    params = model_cfg.get("params", {})
    
    # Standardize dictionary from DictConfig
    params_dict = dict(params) if params is not None else {}

    # Dynamically import the class
    module_name, class_name_str = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name_str)

    print(f"Training {class_name} with parameters: {params_dict}")
    model = model_class(**params_dict)
    model.fit(X_train, y_train)

    return model
