"""Summarize complete validation trajectories from the environment."""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {
    "step",
    "traj_uid",
    "turn_step",
    "data_source",
    "task_index",
    "episode_rewards",
    "tool_callings",
    "projected_action",
    "is_action_valid",
}


def read_turns(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*_batch_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No validation trajectories in {directory}")
    turns = pd.concat((pd.read_json(path, lines=True) for path in files), ignore_index=True)
    missing = REQUIRED.difference(turns.columns)
    if missing:
        raise ValueError(f"Validation trajectories lack fields: {sorted(missing)}")
    return turns.sort_values(["step", "traj_uid", "turn_step"])


def summarize(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    turns = read_turns(directory)
    actions = turns["projected_action"].fillna("").astype(str)
    turns["valid_action"] = turns["is_action_valid"].astype(bool).astype(int)
    turns["invalid_action"] = 1 - turns["valid_action"]
    turns["search_action"] = actions.str.startswith("<search>").astype(int)
    turns["answer_action"] = actions.str.startswith("<answer>").astype(int)

    keys = ["step", "traj_uid"]
    aggregation = {
        "data_source": ("data_source", "first"),
        "task_index": ("task_index", "first"),
        "episode_reward": ("episode_rewards", "last"),
        "turns": ("turn_step", "count"),
        "tool_calls": ("tool_callings", "last"),
        "valid_actions": ("valid_action", "sum"),
        "invalid_actions": ("invalid_action", "sum"),
        "searches": ("search_action", "sum"),
        "answers": ("answer_action", "sum"),
    }
    episodes = turns.groupby(keys, as_index=False).agg(**aggregation)
    episodes["success"] = episodes["episode_reward"] > 0

    summary_aggregation = {
        "episodes": ("traj_uid", "count"),
        "mean_reward": ("episode_reward", "mean"),
        "success_rate": ("success", "mean"),
        "mean_turns": ("turns", "mean"),
        "mean_tool_calls": ("tool_calls", "mean"),
        "mean_searches": ("searches", "mean"),
        "mean_answers": ("answers", "mean"),
        "invalid_actions": ("invalid_actions", "sum"),
    }
    by_source = episodes.groupby(["step", "data_source"], as_index=False).agg(**summary_aggregation)
    overall = episodes.groupby("step", as_index=False).agg(**summary_aggregation)
    overall.insert(1, "data_source", "all")
    summary = pd.concat((overall, by_source), ignore_index=True).sort_values(["step", "data_source"])
    return episodes, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", type=Path)
    args = parser.parse_args()
    episodes, summary = summarize(args.evaluation_dir)
    episodes.to_csv(args.evaluation_dir / "episodes.csv", index=False)
    summary.to_csv(args.evaluation_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
