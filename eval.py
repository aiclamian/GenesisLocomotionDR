import genesis as gs
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

from env import Go2Env


def main():
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)

    gs.init(backend=gs.constants.backend.cpu)
    env = Go2Env(num_envs=1, show_viewer=True, env_cfg=cfg["env"])

    log_dir: str = cfg["run"]["log_dir"]

    runner = OnPolicyRunner(env, cfg["train"], log_dir, gs.device)  # type: ignore
    num_learning_iterations: int = cfg["run"]["num_learning_iterations"]
    runner.load(log_dir + f"/model_{num_learning_iterations}.pt")
    policy = runner.get_inference_policy(device=gs.device)  # type: ignore

    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, *_ = env.step(actions)


if __name__ == "__main__":
    main()
