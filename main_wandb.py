
from easydict import EasyDict as edict
from pathlib import Path
from prettyprinter import pprint
from imageio import imread, imsave
import os, json

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./config", config_name="config_camvid")
def run_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.init(config=cfg, project=cfg.project.name, name=cfg.experiment.name)
    # project_dir = Path(hydra.utils.get_original_cwd())

    from experiments.seg_model import SegModel
    model = SegModel(cfg)
    model.run()


if __name__ == "__main__":
    run_app()
