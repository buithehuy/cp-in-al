
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def verify(cfg: DictConfig):
    print(f"init_epochs: {cfg.trainer.init_epochs}")
    print(f"epochs_per_round: {cfg.trainer.epochs_per_round}")

if __name__ == "__main__":
    verify()
