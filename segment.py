import hydra

from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="segment")
def segment(config: DictConfig) -> None:
    pass