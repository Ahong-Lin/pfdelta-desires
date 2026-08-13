import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.utils.flags import flags
from core.utils.main_utils import load_config, load_registry
from core.utils.registry import registry


def main():
    tic = time.time()
    print("Welcome! Loading lab for fine-tuning...")

    load_registry()

    args, override_args = flags.parser.parse_known_args()
    config = load_config(args, override_args)
    assert config["config_type"] == "experiment", "Fine-tune script expects one experiment config."

    trainer_name = config["functional"].get("trainer_name", "base_trainer")
    trainer_class = registry.get_trainer_class(trainer_name)
    trainer = trainer_class(config)

    init_model = config["functional"].get("init_model")
    assert init_model, "Config must set functional.init_model to a model.pt path."
    if not os.path.exists(init_model):
        raise FileNotFoundError(f"Initial model not found: {init_model}")

    state_dict = torch.load(init_model, map_location=trainer.device)
    trainer.model.load_state_dict(state_dict)
    print(f"Loaded initial model weights from {init_model}")

    print("Starting fine-tuning...")
    trainer.train()

    toc = time.time()
    print(f"Fine-tune script took {toc - tic} seconds to run.")
    print("Closing lab. Goodbye!")
    return trainer


if __name__ == "__main__":
    main()
