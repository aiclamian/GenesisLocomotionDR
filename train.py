import os
import shutil

import genesis as gs
import yaml
from rsl_rl.runners import OnPolicyRunner

from env import Go2Env


def main() -> None:
    with open("cfg.yml", "r") as f:
        cfg: dict[str, dict] = yaml.safe_load(f)

    gs.init(
        backend=gs.constants.backend.amdgpu,
        logging_level="warning",
        seed=2233,
        performance_mode=True,
    )
    env = Go2Env(num_envs=1024, show_viewer=False, env_cfg=cfg["env"])

    log_dir: str = cfg["run"]["log_dir"]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)

    runner = OnPolicyRunner(env, cfg["train"], log_dir, gs.device)  # type: ignore
    num_learning_iterations: int = cfg["run"]["num_learning_iterations"]
    runner.learn(
        num_learning_iterations=num_learning_iterations + 1, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()
