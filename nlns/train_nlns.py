from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from core.baseline import build_baseline_solution
from core.constraints import check_solution_feasibility
from core.evaluation import compute_distance_matrix, compute_objective
from core.model import Route, Solution
from core.parser import parse_instances_dir

from alns.destroy import destroy_solution
from alns.repair import repair_solution
from nlns.policy import ACTION_SPACE, OperatorPolicyNet


def route_load(route: List[int], instance) -> float:
    return sum(instance.all_nodes[n].demand for n in route if n != 0)


def compute_load_variance(routes: List[List[int]], instance) -> float:
    loads = [route_load(route, instance) for route in routes if len(route) > 2]
    if not loads:
        return 0.0
    mean = sum(loads) / len(loads)
    return sum((x - mean) ** 2 for x in loads) / len(loads)


def compute_spatial_variance(routes: List[List[int]], instance) -> float:
    centroids = []
    for route in routes:
        pts = [(instance.all_nodes[n].x, instance.all_nodes[n].y) for n in route if n != 0]
        if not pts:
            continue
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        centroids.append((cx, cy))
    if not centroids:
        return 0.0
    mx = sum(x for x, _ in centroids) / len(centroids)
    my = sum(y for _, y in centroids) / len(centroids)
    return sum((x - mx) ** 2 + (y - my) ** 2 for x, y in centroids) / len(centroids)


def solution_from_baseline(baseline) -> Solution:
    sol = Solution()
    for route in baseline.routes:
        sol.routes.append(Route(path=list(route)))
    sol.unserved_customers = []
    return sol


def extract_state(
    instance,
    solution: Solution,
    distance_matrix,
    step: int,
    max_steps: int,
    baseline_objective: float,
) -> torch.Tensor:
    routes = solution.route_paths
    feas = check_solution_feasibility(routes, instance, distance_matrix)

    if feas.feasible:
        objective, components = compute_objective(routes, instance, distance_matrix)
        total_distance = components["total_distance"]
        total_time = components["total_time"]
    else:
        objective = baseline_objective * 2.0
        total_distance = feas.total_distance
        total_time = feas.total_time

    num_customers = max(1, instance.num_nodes - 1)
    served = set(solution.all_customers())
    unserved = num_customers - len(served)

    routes_used = len(solution.active_routes())
    avg_route_len = sum(max(0, len(r.path) - 2) for r in solution.active_routes()) / max(1, routes_used)
    avg_load = sum(route_load(r.path, instance) for r in solution.active_routes()) / max(1, routes_used)

    state = torch.tensor(
        [
            step / max(1, max_steps),
            len(routes) / max(1, instance.num_vehicles),
            routes_used / max(1, instance.num_vehicles),
            unserved / max(1, num_customers),
            objective / max(1.0, baseline_objective),
            total_distance / max(1.0, baseline_objective),
            total_time / max(1.0, baseline_objective),
            compute_load_variance(routes, instance) / max(1.0, instance.capacity ** 2),
            compute_spatial_variance(routes, instance) / 10000.0,
            avg_route_len / max(1.0, num_customers),
            avg_load / max(1.0, instance.capacity),
            1.0 if feas.feasible else 0.0,
        ],
        dtype=torch.float32,
    )
    return state


def discounted_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    out = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_checkpoint(
    path: Path,
    policy: OperatorPolicyNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_avg_reward: float,
) -> None:
    payload = {
        "epoch": epoch,
        "best_avg_reward": best_avg_reward,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "action_space": ACTION_SPACE,
        "input_dim": 12,
        "hidden_dim": 128,
    }
    torch.save(payload, path)


def train(
    instances_dir: str,
    epochs: int,
    steps_per_episode: int,
    lr: float,
    seed: int,
    save_dir: str,
    gamma: float,
    entropy_beta: float,
    min_remove: int,
    max_remove: int,
    checkpoint_every: int = 1,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    instances = parse_instances_dir(instances_dir)
    if not instances:
        raise FileNotFoundError(f"No instances found in {instances_dir}")

    save_root = _ensure_dir(Path(save_dir))
    checkpoints_dir = _ensure_dir(save_root / "checkpoints")
    logs_dir = _ensure_dir(save_root / "logs")

    rewards_csv_path = logs_dir / "episode_rewards.csv"
    model = OperatorPolicyNet(input_dim=12, hidden_dim=128, n_actions=len(ACTION_SPACE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_avg_reward = float("-inf")
    best_checkpoint_path = checkpoints_dir / "best_model.pt"
    final_checkpoint_path = checkpoints_dir / "final_model.pt"

    with open(rewards_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "episode",
                "instance",
                "loss",
                "episode_reward",
                "baseline_objective",
                "final_objective",
                "feasible",
                "routes_used",
                "steps",
            ],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_reward = 0.0
            used_episodes = 0

            for episode_idx, instance in enumerate(instances, 1):
                baseline = build_baseline_solution(instance)
                if not baseline.feasible:
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "episode": episode_idx,
                            "instance": instance.name,
                            "loss": "",
                            "episode_reward": "",
                            "baseline_objective": baseline.objective,
                            "final_objective": "",
                            "feasible": False,
                            "routes_used": "",
                            "steps": 0,
                        }
                    )
                    continue

                solution = solution_from_baseline(baseline)
                distance_matrix = compute_distance_matrix(instance)
                base_obj = baseline.objective

                log_probs = []
                values = []
                entropies = []
                rewards = []

                current_obj = base_obj
                final_feasible = True

                for step in range(steps_per_episode):
                    state = extract_state(
                        instance=instance,
                        solution=solution,
                        distance_matrix=distance_matrix,
                        step=step,
                        max_steps=steps_per_episode,
                        baseline_objective=base_obj,
                    )
                    action, log_prob, value, entropy = model.act(state)

                    destroy_name, repair_name = ACTION_SPACE[int(action.item())]

                    num_remove = random.randint(min_remove, max_remove)
                    destroy_result = destroy_solution(
                        solution=solution,
                        instance=instance,
                        num_remove=num_remove,
                        method=destroy_name,
                        distance_matrix=distance_matrix,
                        rng=random,
                    )

                    repair_result = repair_solution(
                        solution=destroy_result.partial_solution,
                        removed_customers=destroy_result.removed_customers,
                        instance=instance,
                        method=repair_name,
                        distance_matrix=distance_matrix,
                        rng=random,
                    )

                    candidate = repair_result.repaired_solution
                    candidate_routes = candidate.route_paths
                    feas = check_solution_feasibility(candidate_routes, instance, distance_matrix)

                    if feas.feasible:
                        cand_obj, _ = compute_objective(candidate_routes, instance, distance_matrix)
                        reward = (current_obj - cand_obj) / max(1.0, current_obj)
                        if cand_obj < current_obj:
                            solution = candidate.copy()
                            current_obj = cand_obj
                    else:
                        reward = -1.0
                        final_feasible = False

                    reward = float(max(-5.0, min(5.0, reward)))

                    log_probs.append(log_prob)
                    values.append(value)
                    entropies.append(entropy)
                    rewards.append(reward)

                if not rewards:
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "episode": episode_idx,
                            "instance": instance.name,
                            "loss": "",
                            "episode_reward": "",
                            "baseline_objective": base_obj,
                            "final_objective": "",
                            "feasible": False,
                            "routes_used": "",
                            "steps": steps_per_episode,
                        }
                    )
                    continue

                returns = discounted_returns(rewards, gamma=gamma)
                values_t = torch.stack(values)
                log_probs_t = torch.stack(log_probs)
                entropies_t = torch.stack(entropies)

                advantages = returns - values_t.detach()
                policy_loss = -(log_probs_t * advantages).mean()
                value_loss = F.mse_loss(values_t, returns)
                entropy_loss = -entropy_beta * entropies_t.mean()
                loss = policy_loss + 0.5 * value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                episode_reward = float(sum(rewards))
                epoch_loss += float(loss.item())
                epoch_reward += episode_reward
                used_episodes += 1

                writer.writerow(
                    {
                        "epoch": epoch,
                        "episode": episode_idx,
                        "instance": instance.name,
                        "loss": float(loss.item()),
                        "episode_reward": episode_reward,
                        "baseline_objective": base_obj,
                        "final_objective": current_obj,
                        "feasible": final_feasible,
                        "routes_used": len(solution.active_routes()),
                        "steps": steps_per_episode,
                    }
                )

            avg_loss = epoch_loss / max(1, used_episodes)
            avg_reward = epoch_reward / max(1, used_episodes)

            print(
                f"Epoch {epoch}/{epochs} | episodes={used_episodes} | loss={avg_loss:.4f} | reward={avg_reward:.4f}",
                flush=True,
            )

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                _save_checkpoint(best_checkpoint_path, model, optimizer, epoch, best_avg_reward)

            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                epoch_ckpt = checkpoints_dir / f"epoch_{epoch}.pt"
                _save_checkpoint(epoch_ckpt, model, optimizer, epoch, best_avg_reward)

        _save_checkpoint(final_checkpoint_path, model, optimizer, epochs, best_avg_reward)

    print(f"Saved reward log to: {rewards_csv_path}", flush=True)
    print(f"Saved best checkpoint to: {best_checkpoint_path}", flush=True)
    print(f"Saved final checkpoint to: {final_checkpoint_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deep-RL policy to guide ALNS operator choice.")
    parser.add_argument("--instances_dir", required=True, help="TXT instances directory, e.g. data/train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_episode", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy_beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_remove", type=int, default=5)
    parser.add_argument("--max_remove", type=int, default=20)
    parser.add_argument("--save_dir", default="outputs/nlns", help="Directory for checkpoints and logs")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save epoch checkpoints every N epochs")
    args = parser.parse_args()

    train(
        instances_dir=args.instances_dir,
        epochs=args.epochs,
        steps_per_episode=args.steps_per_episode,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
        gamma=args.gamma,
        entropy_beta=args.entropy_beta,
        min_remove=args.min_remove,
        max_remove=args.max_remove,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()