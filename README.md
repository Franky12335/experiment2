# SDRD: Main SearchQA Experiments

This repository provides training and evaluation for Qwen2.5-3B-Instruct and Qwen3-1.7B. The method coefficient lambda is 0.02.

## Method

The policy generates queries or answers and receives retrieval results. During training, reference answers, retrieved passages, and selected skills provide additional teacher input. The teacher samples candidate content while retaining the original response prefixes. Candidate weights are computed from teacher and original policy scores and held fixed during the policy update. The policy learns from these candidates together with GRPO. Candidate counts, teacher input construction, and averaging rules are defined in configs/method.yaml.

## Datasets

The complete datasets are included under data/searchqa/processed/ and require no further preprocessing.

| File | Examples | Purpose |
| --- | ---: | --- |
| train.parquet | 169,615 | Training on NQ and HotpotQA |
| test.parquet | 51,713 | Evaluation on seven datasets |

The test split covers NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, and Bamboogle. The training entry point reads both files automatically. Reference answers are used for training guidance and evaluation scoring; they are not added to the ordinary policy input. Data originate from the [public Search-R1 dataset](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train).

Wikipedia passages and the retrieval index are prepared separately using the download script.

## Dependencies and Resources

Use Linux, Python 3.12, and CUDA 12.8. Training uses four 80 GB GPUs. The retrieval encoder uses an additional GPU, while the FAISS index runs on the CPU. At least 128 GB of host RAM is recommended.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -r requirements-kernels.txt --no-build-isolation
python -m pip install -e . --no-deps

python scripts/download_resources.py --models all
python scripts/download_resources.py --retrieval
~~~

Installing FlashAttention requires the CUDA toolkit and a C++ compiler. Resources default to models/ and data/searchqa/. SDRD_MODEL_DIR and SDRD_DATA_DIR can select existing complete resource directories. SDRD_OUTPUT_DIR can select an output directory.

## Training

Start the retrieval service first:

~~~bash
python scripts/start_retriever.py --gpu 0
~~~

In another terminal, select one model and start training:

~~~bash
python scripts/run_experiment.py --model qwen2.5_3b --devices 1,2,3,4
python scripts/run_experiment.py --model qwen3_1.7b --devices 1,2,3,4
~~~

Training uses seed 42, 150 optimizer updates, a learning rate of 1e-6, 15 warmup updates, and eight responses per question. Detailed settings are in configs/. Each run records its configuration, log, checkpoints, and evaluation outputs under outputs/MODEL/RUN/.

## Evaluation

Replace RUN with the actual training directory:

~~~bash
python -m web_agent.evaluate --model qwen2.5_3b --checkpoint outputs/qwen2.5_3b/RUN/checkpoints/global_step_150 --devices 1,2,3,4
~~~

For Qwen3-1.7B, select the corresponding model and checkpoint. Evaluation restores the model and runs the complete test split, writing results to a new run directory. Summarize its evaluation trajectories with:

~~~bash
python -m web_agent.analyze outputs/qwen2.5_3b/EVAL_RUN/evaluation
~~~

The method implementation is in web_agent/. The SearchQA environment is in agent_system/, distributed training components are in verl/, the retrieval service is in search_r1/, and retrieval skills are in skills/. Third-party licenses are retained in LICENSE and licenses/.
