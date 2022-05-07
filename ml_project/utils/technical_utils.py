import importlib
from pathlib import Path
from typing import Any, Union

from ml_project.entities import Config


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def get_last_artifacts_path(cfg: Config) -> Union[Path, str]:
    project_dir = Path(cfg.general.project_dir)
    artifacts_dir = project_dir / cfg.general.artifacts_dir

    folders = sorted(
        list(
            filter(
                lambda p: "train_pipeline.log"
                in list(map(lambda x: x.name, p.iterdir())),
                artifacts_dir.iterdir(),
            )
        )
    )
    if folders:
        last_artifacts_path = artifacts_dir / folders[-1]
    else:
        last_artifacts_path = ""
    return last_artifacts_path
