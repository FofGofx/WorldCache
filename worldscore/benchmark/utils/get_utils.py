#WorldScore/worldscore/benchmark/utils/get_utils.py
import importlib

from worldscore.benchmark.utils.utils import get_model2type, type2model


def get_adapter_function(model_type, model):
    module_name = f"worldscore.benchmark.helpers.adapters.{model_type}.adapter_{model}"
    function_name = f"adapter_{model}"

    adapter_module = importlib.import_module(module_name)
    adapter_function = getattr(adapter_module, function_name)
    return adapter_function


def get_adapter(config):
    model = config["model"]
    model_type = get_model2type(type2model)[model]

    if model_type == "videogen":
        model = config['generate_type']  
    try:
        return get_adapter_function(model_type, model)
    except Exception as e:
        raise AttributeError(  # pylint: disable=raise-missing-from  # noqa: B904
            f"Failed to import adapter for {model} - {model_type}: {e}"
        )


def get_dataloader(config):
    model = config["model"]
    model_type = get_model2type(type2model)[model]
    module_name = "worldscore.benchmark.helpers.dataloaders"

    try:
        return getattr(
            importlib.import_module(module_name), f"dataloader_{model_type}"
        )(config)
    except Exception as e:
        raise AttributeError(f"Failed to import dataLoader for {model_type}: {e}")  # pylint: disable=raise-missing-from  # noqa: B904
