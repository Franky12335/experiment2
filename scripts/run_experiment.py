"""Train or evaluate the main SearchQA method."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with (ROOT / "configs" / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resource_roots():
    return {
        key: Path(os.environ.get(variable, ROOT / default)).expanduser().resolve()
        for key, variable, default in (
            ("data", "SDRD_DATA_DIR", "data"),
            ("models", "SDRD_MODEL_DIR", "models"),
            ("output", "SDRD_OUTPUT_DIR", "outputs"),
        )
    }


def resolve_settings(model_name):
    task = load("searchqa.yaml")["task"]
    model = load("model.yaml")["model"]["backbones"][model_name]
    document = load("method.yaml")
    method = {**document["method"], **document["profiles"][model_name]}
    optimization = load("training_optimization.yaml")["training_optimization"]
    coefficient = method["guidance_coef"]
    if not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or coefficient != 0.02:
        raise ValueError("The method requires guidance_coef=0.02.")
    if method["gate_beta"] != 1.0 or method["num_content_samples"] not in (1, 4):
        raise ValueError("Invalid method settings.")
    if optimization["resume_mode"] != "disable" or optimization["seed"] != 42:
        raise ValueError("Training requires seed 42 and a fresh initialization.")
    return task, model, method, optimization


def check_resources(model_name, roots):
    task, model, method, _ = resolve_settings(model_name)
    required = [roots["data"] / task[key] for key in ("train_file", "val_file")]
    model_dir = roots["models"] / model["directory"]
    required += [model_dir / name for name in ("config.json", "tokenizer_config.json", "tokenizer.json")]
    weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No model weights in {model_dir}")
    required += weights
    required += [ROOT / method["skills"]["directory"] / "search" / name
                 for name in ("claude_style_skills_search.json", "skill_mapping.json")]
    retrieval = load("retrieval.yaml")["retrieval"]
    required += [roots["data"] / retrieval[key] for key in ("index", "corpus")]
    required += [roots["models"] / retrieval["encoder"] / name
                 for name in ("config.json", "tokenizer.json")]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing resources:\n" + "\n".join(missing))
    url = f"http://{retrieval['host']}:{retrieval['port']}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=15) as response:
            health = json.load(response)
    except OSError as exc:
        raise RuntimeError("Start scripts/start_retriever.py before training or evaluation.") from exc
    if health.get("status") != "ready":
        raise RuntimeError("The retrieval service is not ready.")
    for key, reported in (("index", "index_path"), ("corpus", "corpus_path")):
        if Path(health[reported]).resolve() != (roots["data"] / retrieval[key]).resolve():
            raise ValueError(f"Retrieval {key} differs from the configured resource.")
    return len(required)


def build_run(args):
    roots = resource_roots()
    task_name = "searchqa"
    task, model, method, opt = resolve_settings(args.model)
    train = load("training.yaml")["training"]
    backend = load("verl.yaml")["verl"]
    rollout = load("vllm.yaml")["vllm"]
    evaluation = load("evaluation.yaml")["evaluation"]
    settings = opt["task_config"]["searchqa"]
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run-name must be one directory name.")
    run_dir = roots["output"] / args.model / run_name
    steps, group_size = opt["total_training_steps"], opt["group_size"]
    val_batch, max_turns = evaluation["val_batch_size"], task["max_turns"]
    response_length = train["max_response_length"]
    if args.devices:
        devices = args.devices.split(",")
        if (not all(value.strip().isdigit() for value in devices)
                or len(set(value.strip() for value in devices)) != len(devices)
                or len(devices) != settings["n_gpus"]):
            raise ValueError(f"This experiment needs {settings['n_gpus']} distinct GPU indices.")
    checkpoints, val_dir = run_dir / "checkpoints", run_dir / "evaluation"
    values = {
        "hydra.run.dir": str(run_dir / "hydra"),
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "critic.strategy": "fsdp2",
        "actor_rollout_ref.ref.strategy": "fsdp2",
        "data.train_files": str(roots["data"] / task["train_file"]),
        "data.val_files": str(roots["data"] / task["val_file"]),
        "data.train_batch_size": settings["train_batch_size"],
        "data.val_batch_size": val_batch,
        "data.max_prompt_length": train["max_prompt_length"],
        "data.max_response_length": response_length,
        "data.filter_overlong_prompts": True,
        "data.truncation": "left",
        "data.return_raw_chat": True,
        "actor_rollout_ref.model.path": str(roots["models"] / model["directory"]),
        "actor_rollout_ref.actor.optim.lr": opt["learning_rate"],
        "actor_rollout_ref.actor.optim.lr_warmup_steps": opt["lr_warmup_steps"],
        "actor_rollout_ref.actor.optim.warmup_style": "constant",
        "actor_rollout_ref.actor.optim.weight_decay": opt["weight_decay"],
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": settings["ppo_mini_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": settings["ppo_micro_batch_size_per_gpu"],
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": settings["kl_loss_coef"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.grad_clip": opt["grad_clip"],
        "actor_rollout_ref.actor.entropy_coeff": opt["entropy_coef"],
        "actor_rollout_ref.actor.clip_ratio": opt["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_low": opt["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_high": opt["clip_ratio"],
        "actor_rollout_ref.actor.use_invalid_action_penalty": True,
        "actor_rollout_ref.actor.invalid_action_penalty_coef": settings["invalid_action_penalty_coef"],
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.dtype": "bfloat16",
        "actor_rollout_ref.rollout.temperature": opt["train_temperature"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": rollout["tensor_model_parallel_size"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": rollout["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": rollout["log_prob_micro_batch_size_per_gpu"],
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": rollout["log_prob_max_token_len_per_gpu"],
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": rollout["log_prob_micro_batch_size_per_gpu"],
        "env.env_name": task["environment"],
        "env.seed": opt["seed"],
        "env.max_steps": max_turns,
        "env.history_length": task["history_length"],
        "env.rollout.n": group_size,
        "trainer.critic_warmup": 0,
        "trainer.logger": ["console"],
        "trainer.project_name": "anonymous",
        "trainer.experiment_name": f"{task_name}_{args.model}_{run_name}",
        "trainer.n_gpus_per_node": settings["n_gpus"],
        "trainer.nnodes": 1,
        "trainer.ray_wait_register_center_timeout": 600,
        "trainer.default_local_dir": str(checkpoints),
        "trainer.validation_data_dir": str(val_dir),
        "trainer.rollout_data_dir": str(run_dir / "rollouts"),
        "trainer.save_freq": backend["save_freq"],
        "trainer.test_freq": evaluation["test_freq"],
        "trainer.val_before_train": evaluation["val_before_train"],
        "trainer.total_training_steps": steps,
        "trainer.total_epochs": opt["epoch_cap"],
        "trainer.resume_mode": "disable",
        "+data.seed": opt["seed"],
        "+actor_rollout_ref.actor.training_seed": opt["seed"],
        "+actor_rollout_ref.rollout.seed": opt["seed"],
        "+actor_rollout_ref.actor.optim.betas": opt["adamw_betas"],
        "+algorithm.obs_distill.task": task_name,
        "+algorithm.obs_distill.guidance_coef": method["guidance_coef"],
        "+algorithm.obs_distill.gate_beta": method["gate_beta"],
        "+algorithm.obs_distill.num_content_samples": method["num_content_samples"],
        "+algorithm.obs_distill.skills.directory": str(ROOT / method["skills"]["directory"]),
        "+algorithm.obs_distill.skills.max_skills": method["skills"]["max_skills"],
        "+algorithm.obs_distill.training_settings": opt,
    }

    for name in ("normalization", "teacher_context"):
        values["+algorithm.obs_distill." + name] = method[name]
    retrieval = load("retrieval.yaml")["retrieval"]
    values.update({
        "env.search.search_url": f"http://{retrieval['host']}:{retrieval['port']}/retrieve",
        "env.search.topk": retrieval["topk"],
        "+env.search.action_budget_hint": method["action_budget_hint"],
    })
    if model["name"] == "qwen3":
        values["+data.apply_chat_template_kwargs.enable_thinking"] = False
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.name.startswith("global_step_") or not checkpoint.name.removeprefix("global_step_").isdigit():
            raise ValueError("checkpoint must name a global_step_NUMBER directory.")
        if not (checkpoint / "actor").is_dir():
            raise FileNotFoundError(f"Expected a checkpoint containing actor/: {checkpoint}")
        values.update({
            "trainer.val_only": True, "trainer.val_before_train": True,
            "trainer.resume_mode": "resume_path", "trainer.resume_from_path": str(checkpoint),
            "trainer.default_local_dir": str(checkpoint.parent),
            "trainer.save_freq": -1, "trainer.test_freq": -1,
        })
    command = [sys.executable, "-m", "web_agent.training.main"]
    for key, value in values.items():
        encoded = (yaml.safe_dump(value, default_flow_style=True, sort_keys=False, width=sys.maxsize).strip()
                   if isinstance(value, dict) else json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        command.append(f"{key}={encoded}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    for key, variable in (("data", "SDRD_DATA_DIR"), ("models", "SDRD_MODEL_DIR"), ("output", "SDRD_OUTPUT_DIR")):
        environment[variable] = str(roots[key])
    environment["PYTHONHASHSEED"] = str(opt["seed"])
    if args.devices:
        environment["CUDA_VISIBLE_DEVICES"] = args.devices
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[key] = "1"
    environment["RAY_TMPDIR"] = str(Path(tempfile.gettempdir()) / f"sd_{os.getpid()}")
    return roots, method, run_dir, command, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen2.5_3b", "qwen3_1.7b"), required=True)
    parser.add_argument("--devices", help="Four visible GPU indices.")
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint", help="Evaluate a saved global_step directory.")
    args = parser.parse_args()
    roots, method, run_dir, command, environment = build_run(args)
    check_resources(args.model, roots)
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {"model": args.model, "method": method,
              "mode": "evaluation" if args.checkpoint else "training", "command": command}
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "training.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    record["exit_code"] = result.returncode
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Run output: {run_dir}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
