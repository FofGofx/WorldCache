#WorldCache/worldscore/benchmark/helpers/__init__.py
import os

from omegaconf import OmegaConf

from worldscore.benchmark.helpers.helper import Helper
from worldscore.benchmark.utils.get_utils import get_dataloader
from worldscore.benchmark.helpers.dataloaders import dataloader_general
from worldscore.benchmark.utils.utils import check_model

def GetHelpers(model_name, visual_movement, json_file="", **kwargs):
    assert check_model(model_name), 'Model not exists!'
    worldcache_output_dir = kwargs.get("worldcache_output_dir")

    ### Get dataloader, helper for model runing
    if json_file == "" or json_file is None:
        json_file = f"{visual_movement}.json"

    root_path = os.getenv("WORLDSCORE_PATH", "/data4/fanguoxin/WorldCache")
    dataset_root = os.getenv("DATA_PATH", "/data4/fanguoxin/WorldCache/data")
    json_path = os.path.join(dataset_root, "WorldScore-Dataset", visual_movement, json_file)

    base_config = OmegaConf.load(f"{root_path}/config/base_config.yaml")
    config = OmegaConf.load(
        os.path.join(
            f"{root_path}/config/model_configs", f"{model_name}.yaml"
        )
    )
    config = OmegaConf.merge(base_config, config)
    config.json_path = json_path
    config.visual_movement = visual_movement
    # Interpolate environment variables in the YAML file
    config = OmegaConf.to_container(config, resolve=True)

    # 应用 Worldcache 输出目录（显式传递优先，否则回退到环境变量）
    if worldcache_output_dir:
        config['output_dir'] = worldcache_output_dir
    else:
        worldcache_suffix = os.environ.get('WORLDCACHE_OUTPUT_SUFFIX', '').strip()
        if worldcache_suffix:
            if worldcache_suffix.startswith('_'):
                config['output_dir'] = worldcache_suffix[1:]
            else:
                config['output_dir'] = config['output_dir'] + worldcache_suffix

    loader = get_dataloader(config)
    helper = Helper(config)
    dataloader = loader.data

    return dataloader, helper


def GetDataloader(visual_movement, json_file=None, noise=False, noise_type="simple"):
    ### Get dataloader for data analysis
    if json_file is None:
        json_file = f"{visual_movement}.json"

    root_path = os.getenv("WORLDSCORE_PATH", "/data4/fanguoxin/WorldCache")
    dataset_root = os.getenv("DATA_PATH", "/data4/fanguoxin/WorldCache/data")

    json_path = os.path.join(dataset_root, "WorldScore-Dataset", visual_movement, json_file)

    config = OmegaConf.load(f"{root_path}/WorldCache/config/base_config.yaml")

    config.json_path = json_path
    config.visual_movement = visual_movement
    config.noise = noise
    config.noise_type = noise_type
    # Interpolate environment variables in the YAML file
    config = OmegaConf.to_container(config, resolve=True)

    loader = dataloader_general(config)

    return loader.data

