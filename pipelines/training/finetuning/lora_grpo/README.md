# Lora Grpo Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

LoRA GRPO Training Pipeline — RLVR fine-tuning with promotion gating.

A 5-stage pipeline that trains a language model using LoRA GRPO (Group Relative Policy Optimization) for tool-calling tasks, evaluates training results, and conditionally registers and deploys the model only if training shows improvement.

Prerequisites: - A ReadWriteMany PVC (e.g., NFS-backed) for model storage and serving. Create with: ``oc create pvc <name> --access-mode=ReadWriteMany --storage-class=nfs-csi --size=50Gi`` - ``kubernetes-credentials`` secret with KUBERNETES_SERVER_URL and KUBERNETES_AUTH_TOKEN (required for TrainJob
creation). - ``hf-token`` secret (optional, for gated HuggingFace models/datasets). - Pipeline ServiceAccount RBAC for KServe CRDs (required for stage 5).

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_00_infra_man_pvc_name` | `str` | `None` | Name of the user-provided ReadWriteMany PVC. Must exist before pipeline runs. Used for training output, evaluation, and model serving. |
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset URI (hf://, s3://, https://, pvc://). |
| `phase_01_dataset_man_data_split` | `float` | `1.0` | Train/eval split ratio. 1.0 = all for training (default for GRPO — rewards computed at rollout, not from an eval split). |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen3-4B` | Base model (HuggingFace ID or OCI path). |
| `phase_02_train_man_num_iterations` | `int` | `5` | Number of GRPO training iterations. |
| `phase_02_train_man_group_size` | `int` | `4` | Rollouts per prompt for advantage estimation. |
| `phase_02_train_man_prompt_batch_size` | `int` | `50` | Prompts per training batch. |
| `phase_02_train_man_lora_r` | `int` | `16` | LoRA rank (controls adapter capacity). |
| `phase_02_train_man_lora_alpha` | `int` | `8` | LoRA scaling factor. |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip). |
| `phase_04_registry_man_name` | `str` | `grpo-model` | Model name in registry. |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Semantic version (major.minor.patch). |
| `phase_05_deploy_man_namespace` | `str` | `""` | Namespace for KServe deployment. |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit dataset to N samples (0 = all). |
| `phase_02_train_opt_data_path` | `str` | `""` | Override dataset path or HuggingFace ID (bypasses dataset artifact). |
| `phase_02_train_opt_data_config` | `str` | `Qwen3` | HuggingFace dataset config name. |
| `phase_02_train_opt_n_train` | `int` | `200` | Number of training samples from the dataset. |
| `phase_02_train_opt_learning_rate` | `float` | `1e-05` | Learning rate (default 1e-5). |
| `phase_02_train_opt_gpu_memory_utilization` | `float` | `0.45` | Fraction of GPU memory for vLLM rollout inference (rest for training). |
| `phase_02_train_opt_enforce_eager` | `bool` | `True` | Disable torch.compile/CUDAGraphs in vLLM (Qwen3 workaround, default True). |
| `phase_02_train_opt_env_vars` | `str` | `""` | Environment overrides (KEY=VAL,KEY=VAL). |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores per worker. |
| `phase_02_train_opt_gpu` | `int` | `1` | GPUs per worker (should be 1 for ART). |
| `phase_02_train_opt_memory` | `str` | `64Gi` | Memory per worker (e.g., 64Gi). |
| `phase_02_train_opt_labels` | `str` | `""` | Pod labels (key=value,key=value). |
| `phase_02_train_opt_annotations` | `str` | `""` | Pod annotations (key=value,key=value). |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | ClusterTrainingRuntime name. |
| `phase_04_registry_opt_author` | `str` | `pipeline` | Author name for registered model. |
| `phase_04_registry_opt_description` | `str` | `""` | Model description for registry. |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx). |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model Registry server port. |
| `phase_05_deploy_opt_gpu_count` | `int` | `1` | GPUs for the KServe predictor. |
| `phase_05_deploy_opt_max_model_len` | `int` | `4096` | Max context length for vLLM serving. |
| `phase_05_deploy_opt_min_replicas` | `int` | `1` | Minimum predictor replicas. |
| `phase_05_deploy_opt_max_replicas` | `int` | `1` | Maximum predictor replicas. |
| `phase_05_deploy_opt_cpu_requests` | `str` | `2` | CPU for the KServe predictor. |
| `phase_05_deploy_opt_memory_requests` | `str` | `8Gi` | Memory for the KServe predictor. |

## Metadata 🗂️

- **Name**: lora_grpo_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: Kubernetes, Version: >=1.28.0
    - Name: Training Hub, Version: >=0.9.2
    - Name: ART (OpenPipe), Version: >=0.1.0
    - Name: Model Registry, Version: >=0.3.4
    - Name: KServe, Version: >=0.11.0
- **Tags**:
  - training
  - fine_tuning
  - grpo
  - lora
  - peft
  - rlvr
  - reinforcement_learning
  - tool_calling
  - llm
  - pipeline
- **Last Verified**: 2026-09-18 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina
    - JaZeeGH
    - Sridhar1030
  - Reviewers:
    - ChughShilpa
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
