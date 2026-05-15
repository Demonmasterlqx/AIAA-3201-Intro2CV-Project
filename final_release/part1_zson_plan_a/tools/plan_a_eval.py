#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from collections import Counter
from typing import Dict, List

import imageio
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import zson  # noqa: F401
from habitat import make_dataset

from zson.config import get_config
from zson.environment import SimpleRLEnv
from zson.plan_a import ClipDenseEncoder, PlanAAgent, TextGoalCache
from zson.plan_a.reporting import (
    aggregate_metrics,
    build_repo_audit_markdown,
    build_report_markdown,
    classify_failure,
    slugify,
    write_json,
    write_summary_csv,
    write_trace,
)
from zson.plan_a.visualization import (
    annotate_ego,
    compose_frame,
    render_topdown_frame,
    write_video,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing the strict Plan A experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser.parse_args()


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def collect_env_info() -> Dict[str, object]:
    import clip
    import habitat
    import habitat_sim

    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "habitat_version": getattr(habitat, "__version__", "unknown"),
        "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
        "clip_version": getattr(clip, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": current_git_commit(),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_run_dir(config) -> str:
    run_name = config.PLAN_A.RUN_NAME
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.PLAN_A.DATASET}_{config.PLAN_A.SPLIT}_{timestamp}"
    return os.path.join(
        config.PLAN_A.OUTPUT_ROOT,
        run_name,
        config.PLAN_A.DATASET,
        config.PLAN_A.SPLIT,
    )


def dataset_check_payload(config, dataset, selected_episodes: int) -> Dict[str, object]:
    data_path = config.TASK_CONFIG.DATASET.DATA_PATH.format(
        split=config.TASK_CONFIG.DATASET.SPLIT
    )
    content_dir = os.path.join(os.path.dirname(data_path), "content")
    scene_ids = sorted({episode.scene_id for episode in dataset.episodes})
    return {
        "dataset": config.PLAN_A.DATASET,
        "split": config.PLAN_A.SPLIT,
        "data_path": data_path,
        "data_path_exists": os.path.exists(data_path),
        "content_dir": content_dir,
        "content_dir_exists": os.path.isdir(content_dir),
        "scenes_dir": config.TASK_CONFIG.DATASET.SCENES_DIR,
        "scenes_dir_exists": os.path.isdir(config.TASK_CONFIG.DATASET.SCENES_DIR),
        "episode_count_selected": selected_episodes,
        "scene_count_selected": len(scene_ids),
        "scene_ids": scene_ids,
        "shard_index": config.PLAN_A.SHARD_INDEX,
        "num_shards": config.PLAN_A.NUM_SHARDS,
    }


def build_episode_dir(root_dir: str, episode) -> str:
    scene_name = slugify(os.path.basename(episode.scene_id).replace(".glb", ""))
    goal_name = slugify(episode.object_category)
    episode_name = f"{scene_name}_{episode.episode_id}_{goal_name}"
    return os.path.join(root_dir, "episodes", episode_name)


def path_length_from_trace(trace: List[Dict[str, object]]) -> float:
    if len(trace) < 2:
        return 0.0
    total = 0.0
    prev = np.asarray(trace[0]["agent_position"], dtype=np.float32)
    for row in trace[1:]:
        cur = np.asarray(row["agent_position"], dtype=np.float32)
        total += float(np.linalg.norm(cur[[0, 2]] - prev[[0, 2]]))
        prev = cur
    return total


def save_config_snapshot(config, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config_resolved.yaml"), "w") as file:
        file.write(config.dump())


def select_dataset_episodes(dataset, config):
    episodes = list(dataset.episodes)
    if config.PLAN_A.NUM_SHARDS > 1:
        episodes = episodes[
            int(config.PLAN_A.SHARD_INDEX) :: int(config.PLAN_A.NUM_SHARDS)
        ]
    if config.PLAN_A.MAX_EPISODES > 0:
        episodes = episodes[: int(config.PLAN_A.MAX_EPISODES)]
    dataset.episodes = episodes
    return dataset


def maybe_render_frame(agent, observation, info, decision):
    if info["top_down_map"] is None:
        return None, None
    ego = annotate_ego(
        observation[agent.plan_cfg.EGO_SENSOR],
        [
            f"goal: {agent.goal_category}",
            f"step: {agent.step_id}",
            f"action: {decision.action_name}",
            f"reason: {decision.reason}",
            f"mode: {decision.strategy}",
            f"center sim: {decision.center_similarity:.3f}",
            "center depth: {:.3f}".format(decision.center_depth)
            if decision.center_depth is not None
            else "center depth: n/a",
            f"map max sim: {decision.map_max_similarity:.3f}",
        ],
    )
    topdown = render_topdown_frame(
        topdown_info=info["top_down_map"],
        semantic_map=agent.semantic_map,
        trajectory=agent.trajectory,
        current_goal_cell=decision.current_goal_cell,
        output_height=ego.shape[0],
    )
    return compose_frame(ego, topdown), topdown


def run_episode(env, agent, output_dir: str):
    observation = env.reset()
    episode = env.current_episode
    info = env.habitat_env.get_metrics()
    agent.reset(episode, info)
    episode_dir = build_episode_dir(output_dir, episode)
    os.makedirs(episode_dir, exist_ok=True)

    frames = []
    trace = []
    final_topdown = None
    start_time = time.perf_counter()
    done = False
    steps = 0
    final_info = info

    while not done:
        decision = agent.act(observation, info)
        if agent.plan_cfg.WRITE_VIDEOS:
            frame, topdown = maybe_render_frame(agent, observation, info, decision)
            if frame is not None:
                frames.append(frame)
            if topdown is not None:
                final_topdown = topdown

        observation, _, done, final_info = env.step(decision.action)
        trace.append(
            {
                "step": steps,
                "action": decision.action_name,
                "reason": decision.reason,
                "strategy": decision.strategy,
                "center_similarity": decision.center_similarity,
                "center_depth": decision.center_depth,
                "map_max_similarity": decision.map_max_similarity,
                "projected_cells": decision.projected_cells,
                "current_goal_cell": decision.current_goal_cell,
                "current_goal_score": decision.current_goal_score,
                "subgoal_strategy": agent.subgoal_strategy,
                "force_frontier_steps": int(decision.force_frontier_steps),
                "recovery_queue_len": int(decision.recovery_queue_len),
                "stuck": bool(decision.stuck),
                "rotation_stuck": bool(decision.rotation_stuck),
                "agent_position": np.asarray(final_info["agent_position"]).tolist(),
                "distance_to_goal": float(final_info.get("distance_to_goal", 0.0)),
                "success": float(final_info.get("success", 0.0)),
                "spl": float(final_info.get("spl", 0.0)),
            }
        )
        steps += 1
        info = final_info

    runtime_sec = time.perf_counter() - start_time
    action_counts = Counter(row["action"] for row in trace)
    reason_counts = Counter(row["reason"] for row in trace)

    if agent.plan_cfg.WRITE_VIDEOS:
        final_decision = agent.act(observation, final_info)
        frame, topdown = maybe_render_frame(agent, observation, final_info, final_decision)
        if frame is not None:
            frames.append(frame)
        if topdown is not None:
            final_topdown = topdown

    metrics = {
        "scene_id": episode.scene_id,
        "episode_id": episode.episode_id,
        "goal_category": episode.object_category,
        "success": float(final_info.get("success", 0.0)),
        "spl": float(final_info.get("spl", 0.0)),
        "softspl": float(final_info.get("softspl", 0.0)),
        "distance_to_goal": float(final_info.get("distance_to_goal", 0.0)),
        "steps": int(steps),
        "path_length": path_length_from_trace(trace),
        "runtime_sec": float(runtime_sec),
        "timed_out": int(steps) >= int(env.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS),
        "max_similarity_seen": float(agent.max_similarity_seen),
        "failed_goal_count": int(agent.failed_goal_count),
        "stuck_events": int(agent.stuck_events),
        "stop_called": bool(agent.stop_called),
        "semantic_selections": int(agent.semantic_selections),
        "frontier_selections": int(agent.frontier_selections),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "video_path": "",
        "semantic_image_path": "",
    }
    failure_reason = classify_failure(
        metrics,
        semantic_threshold=float(env.config.PLAN_A.SEMANTIC_THRESHOLD),
    )
    metrics["failure_reason"] = failure_reason

    video_path = os.path.join(episode_dir, "ego_topdown.mp4")
    if agent.plan_cfg.WRITE_VIDEOS and len(frames) > 0:
        write_video(frames, video_path, fps=int(agent.plan_cfg.VIDEO_FPS))
        metrics["video_path"] = video_path
    if final_topdown is not None and env.config.PLAN_A.SAVE_DEBUG_IMAGES:
        image_path = os.path.join(episode_dir, "semantic_final.png")
        imageio.imwrite(image_path, final_topdown)
        metrics["semantic_image_path"] = image_path

    write_json(os.path.join(episode_dir, "metrics.json"), metrics)
    write_trace(os.path.join(episode_dir, "trace.jsonl"), trace)
    with open(os.path.join(episode_dir, "failure_reason.txt"), "w") as file:
        file.write(failure_reason + "\n")

    return metrics


def main():
    args = parse_args()
    config = get_config(args.exp_config, args.opts)
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = config.PLAN_A.SPLIT
    config.EVAL.SPLIT = config.PLAN_A.SPLIT
    config.freeze()

    set_seed(int(config.PLAN_A.SEED))
    run_dir = resolve_run_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    save_config_snapshot(config, run_dir)

    env_info = collect_env_info()
    write_json(os.path.join(run_dir, "env_info.json"), env_info)
    with open(os.path.join(run_dir, "repo_audit.md"), "w") as file:
        file.write(build_repo_audit_markdown())

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset = select_dataset_episodes(dataset, config)
    write_json(
        os.path.join(run_dir, "dataset_check.json"),
        dataset_check_payload(config, dataset, len(dataset.episodes)),
    )

    clip_encoder = ClipDenseEncoder(model_name=config.PLAN_A.CLIP_MODEL)
    text_cache = TextGoalCache(
        encoder=clip_encoder,
        prompt_template=config.PLAN_A.PROMPT_TEMPLATE,
        cache_dir=run_dir,
    )

    env = SimpleRLEnv(config, dataset)
    env.seed(int(config.PLAN_A.SEED))
    agent = PlanAAgent(
        sim=env.habitat_env.sim,
        config=config,
        clip_encoder=clip_encoder,
        text_cache=text_cache,
    )

    records = []
    try:
        for _ in range(len(dataset.episodes)):
            metrics = run_episode(env, agent, run_dir)
            records.append(metrics)
    finally:
        text_cache.dump()
        env.close()

    aggregate = aggregate_metrics(records)
    summary_payload = {
        "dataset": config.PLAN_A.DATASET,
        "split": config.PLAN_A.SPLIT,
        "aggregate": aggregate,
        "episodes": records,
    }
    write_json(os.path.join(run_dir, "summary.json"), summary_payload)
    write_summary_csv(os.path.join(run_dir, "summary.csv"), records)
    with open(os.path.join(run_dir, "report.md"), "w") as file:
        file.write(build_report_markdown(config, env_info, aggregate, records))


if __name__ == "__main__":
    main()
