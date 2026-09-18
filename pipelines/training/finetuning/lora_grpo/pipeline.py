"""LoRA GRPO (Group Relative Policy Optimization) Training Pipeline.

A 5-stage pipeline for LoRA GRPO fine-tuning with promotion gating:
1. Dataset Download — validates tool-call schema on CPU
2. LoRA GRPO Training — ART backend, 1 GPU TrainJob, merges LoRA adapters
3. GRPO Evaluation — extracts reward metrics, computes promotion gate
4. Model Registry — registers promoted model (gated by promotion check)
5. Model Deployment — deploys via KServe with vLLM (gated by promotion check)

Uses the Kubeflow Trainer with ART backend on a single GPU TrainJob.
Requires a user-provided ReadWriteMany PVC for persistent model storage
and serving.
"""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.data_processing.dataset_download import dataset_download
from components.deployment.kubeflow_model_registry import kubeflow_model_registry
from components.deployment.model_deployment import model_deployment
from components.evaluation.grpo_eval import grpo_eval
from components.training.finetuning.lora_grpo import train_model

# =============================================================================
# Pipeline Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_MOUNT_PATH = "/mnt/pipeline"
PIPELINE_NAME = "lora-grpo-pipeline"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "LoRA GRPO pipeline: reinforcement learning from verifiable rewards using ART backend on a single GPU TrainJob"
    ),
)
def lora_grpo_pipeline(
    # =========================================================================
    # INFRASTRUCTURE (Required)
    # =========================================================================
    phase_00_infra_man_pvc_name: str,
    # =========================================================================
    # KEY PARAMETERS (Required/Important) — Sorted by stage
    # =========================================================================
    # Stage 1: Dataset
    phase_01_dataset_man_data_uri: str,
    phase_01_dataset_man_data_split: float = 1.0,
    # Stage 2: Training
    phase_02_train_man_model: str = "Qwen/Qwen3-4B",
    phase_02_train_man_num_iterations: int = 5,
    phase_02_train_man_group_size: int = 4,
    phase_02_train_man_prompt_batch_size: int = 50,
    phase_02_train_man_lora_r: int = 16,
    phase_02_train_man_lora_alpha: int = 8,
    # Stage 4: Registry
    phase_04_registry_man_address: str = "",
    phase_04_registry_man_name: str = "grpo-model",
    phase_04_registry_man_version: str = "1.0.0",
    # Stage 5: Deployment
    phase_05_deploy_man_namespace: str = "",
    # =========================================================================
    # OPTIONAL PARAMETERS — Sorted by stage
    # =========================================================================
    # Stage 1
    phase_01_dataset_opt_subset: int = 0,
    # Stage 2
    phase_02_train_opt_data_path: str = "",
    phase_02_train_opt_data_config: str = "Qwen3",
    phase_02_train_opt_n_train: int = 200,
    phase_02_train_opt_learning_rate: float = 1e-5,
    phase_02_train_opt_gpu_memory_utilization: float = 0.45,
    phase_02_train_opt_enforce_eager: bool = True,
    phase_02_train_opt_env_vars: str = "",
    phase_02_train_opt_cpu: str = "4",
    phase_02_train_opt_gpu: int = 1,
    phase_02_train_opt_memory: str = "64Gi",
    phase_02_train_opt_labels: str = "",
    phase_02_train_opt_annotations: str = "",
    phase_02_train_opt_runtime: str = "training-hub",
    # Stage 4
    phase_04_registry_opt_author: str = "pipeline",
    phase_04_registry_opt_description: str = "",
    phase_04_registry_opt_format_name: str = "pytorch",
    phase_04_registry_opt_format_version: str = "1.0",
    phase_04_registry_opt_port: int = 8080,
    # Stage 5
    phase_05_deploy_opt_gpu_count: int = 1,
    phase_05_deploy_opt_max_model_len: int = 4096,
    phase_05_deploy_opt_min_replicas: int = 1,
    phase_05_deploy_opt_max_replicas: int = 1,
    phase_05_deploy_opt_cpu_requests: str = "2",
    phase_05_deploy_opt_memory_requests: str = "8Gi",
):
    """LoRA GRPO Training Pipeline — RLVR fine-tuning with promotion gating.

    A 5-stage pipeline that trains a language model using LoRA GRPO
    (Group Relative Policy Optimization) for tool-calling tasks, evaluates
    training results, and conditionally registers and deploys the model
    only if training shows improvement.

    Prerequisites:
        - A ReadWriteMany PVC (e.g., NFS-backed) for model storage and serving.
          Create with: ``oc create pvc <name> --access-mode=ReadWriteMany
          --storage-class=nfs-csi --size=50Gi``
        - ``kubernetes-credentials`` secret with KUBERNETES_SERVER_URL and
          KUBERNETES_AUTH_TOKEN (required for TrainJob creation).
        - ``hf-token`` secret (optional, for gated HuggingFace models/datasets).
        - Pipeline ServiceAccount RBAC for KServe CRDs (required for stage 5).

    Args:
        phase_00_infra_man_pvc_name: Name of the user-provided ReadWriteMany PVC.
            Must exist before pipeline runs. Used for training output, evaluation,
            and model serving.
        phase_01_dataset_man_data_uri: Dataset URI (hf://, s3://, https://, pvc://).
        phase_01_dataset_man_data_split: Train/eval split ratio. 1.0 = all for
            training (default for GRPO — rewards computed at rollout, not from
            an eval split).
        phase_02_train_man_model: Base model (HuggingFace ID or OCI path).
        phase_02_train_man_num_iterations: Number of GRPO training iterations.
        phase_02_train_man_group_size: Rollouts per prompt for advantage estimation.
        phase_02_train_man_prompt_batch_size: Prompts per training batch.
        phase_02_train_man_lora_r: LoRA rank (controls adapter capacity).
        phase_02_train_man_lora_alpha: LoRA scaling factor.
        phase_04_registry_man_address: Model Registry address (empty = skip).
        phase_04_registry_man_name: Model name in registry.
        phase_04_registry_man_version: Semantic version (major.minor.patch).
        phase_05_deploy_man_namespace: Namespace for KServe deployment.
        phase_01_dataset_opt_subset: Limit dataset to N samples (0 = all).
        phase_02_train_opt_data_path: Override dataset path or HuggingFace ID
            (bypasses dataset artifact).
        phase_02_train_opt_data_config: HuggingFace dataset config name.
        phase_02_train_opt_n_train: Number of training samples from the dataset.
        phase_02_train_opt_learning_rate: Learning rate (default 1e-5).
        phase_02_train_opt_gpu_memory_utilization: Fraction of GPU memory for
            vLLM rollout inference (rest for training).
        phase_02_train_opt_enforce_eager: Disable torch.compile/CUDAGraphs in
            vLLM (Qwen3 workaround, default True).
        phase_02_train_opt_env_vars: Environment overrides (KEY=VAL,KEY=VAL).
        phase_02_train_opt_cpu: CPU cores per worker.
        phase_02_train_opt_gpu: GPUs per worker (should be 1 for ART).
        phase_02_train_opt_memory: Memory per worker (e.g., 64Gi).
        phase_02_train_opt_labels: Pod labels (key=value,key=value).
        phase_02_train_opt_annotations: Pod annotations (key=value,key=value).
        phase_02_train_opt_runtime: ClusterTrainingRuntime name.
        phase_04_registry_opt_author: Author name for registered model.
        phase_04_registry_opt_description: Model description for registry.
        phase_04_registry_opt_format_name: Model format (pytorch, onnx).
        phase_04_registry_opt_format_version: Model format version.
        phase_04_registry_opt_port: Model Registry server port.
        phase_05_deploy_opt_gpu_count: GPUs for the KServe predictor.
        phase_05_deploy_opt_max_model_len: Max context length for vLLM serving.
        phase_05_deploy_opt_min_replicas: Minimum predictor replicas.
        phase_05_deploy_opt_max_replicas: Maximum predictor replicas.
        phase_05_deploy_opt_cpu_requests: CPU for the KServe predictor.
        phase_05_deploy_opt_memory_requests: Memory for the KServe predictor.
    """
    # =========================================================================
    # Stage 1: Dataset Download
    # =========================================================================
    dataset_download_task = dataset_download(
        dataset_uri=phase_01_dataset_man_data_uri,
        pvc_mount_path=PVC_MOUNT_PATH,
        train_split_ratio=phase_01_dataset_man_data_split,
        subset_count=phase_01_dataset_opt_subset,
        dataset_format="tool_call",
        shared_log_file="pipeline_log.txt",
    )
    dataset_download_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(dataset_download_task, "IfNotPresent")

    kfp.kubernetes.use_secret_as_env(
        dataset_download_task,
        secret_name="s3-secret",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        },
        optional=True,
    )

    # =========================================================================
    # Stage 2: LoRA GRPO Training
    # =========================================================================
    training_task = train_model(
        pvc_path=PVC_MOUNT_PATH,
        dataset=dataset_download_task.outputs["train_dataset"],
        training_base_model=phase_02_train_man_model,
        training_data_path=phase_02_train_opt_data_path,
        training_num_iterations=phase_02_train_man_num_iterations,
        training_group_size=phase_02_train_man_group_size,
        training_prompt_batch_size=phase_02_train_man_prompt_batch_size,
        training_n_train=phase_02_train_opt_n_train,
        training_learning_rate=phase_02_train_opt_learning_rate,
        training_gpu_memory_utilization=phase_02_train_opt_gpu_memory_utilization,
        training_enforce_eager=phase_02_train_opt_enforce_eager,
        training_data_config=phase_02_train_opt_data_config,
        training_lora_r=phase_02_train_man_lora_r,
        training_lora_alpha=phase_02_train_man_lora_alpha,
        training_envs=phase_02_train_opt_env_vars,
        training_resource_cpu_per_worker=phase_02_train_opt_cpu,
        training_resource_gpu_per_worker=phase_02_train_opt_gpu,
        training_resource_memory_per_worker=phase_02_train_opt_memory,
        training_metadata_labels=phase_02_train_opt_labels,
        training_metadata_annotations=phase_02_train_opt_annotations,
        training_runtime=phase_02_train_opt_runtime,
        training_pvc_name=phase_00_infra_man_pvc_name,
    )
    training_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(training_task, "IfNotPresent")

    kfp.kubernetes.use_secret_as_env(
        task=training_task,
        secret_name="kubernetes-credentials",
        secret_key_to_env={
            "KUBERNETES_SERVER_URL": "KUBERNETES_SERVER_URL",
            "KUBERNETES_AUTH_TOKEN": "KUBERNETES_AUTH_TOKEN",
        },
        optional=False,
    )

    kfp.kubernetes.use_secret_as_env(
        task=training_task,
        secret_name="oci-pull-secret-model-download",
        secret_key_to_env={"OCI_PULL_SECRET_MODEL_DOWNLOAD": "OCI_PULL_SECRET_MODEL_DOWNLOAD"},
        optional=True,
    )

    # =========================================================================
    # Stage 3: GRPO Evaluation
    # =========================================================================
    grpo_eval_task = grpo_eval(
        training_results_path=f"{PVC_MOUNT_PATH}/checkpoints/training_results.json",
    )
    grpo_eval_task.after(training_task)
    grpo_eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(grpo_eval_task, "IfNotPresent")

    # Attach HF token to tasks that may access gated resources
    for _task in [dataset_download_task, training_task]:
        kfp.kubernetes.use_secret_as_env(
            task=_task,
            secret_name="hf-token",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
            optional=True,
        )

    # Mount user PVC on all tasks that need shared file access
    for _task in [dataset_download_task, training_task, grpo_eval_task]:
        kfp.kubernetes.mount_pvc(
            task=_task,
            pvc_name=phase_00_infra_man_pvc_name,
            mount_path=PVC_MOUNT_PATH,
        )

    # =========================================================================
    # Stages 4 + 5: Gated by promotion check
    # Only run model registration and deployment if training improved
    # (final_reward > initial_reward across iterations).
    # =========================================================================
    with dsl.If(
        grpo_eval_task.outputs["promotion_passed"] == True,  # noqa: E712 — KFP dsl.If requires explicit == comparison
        name="promotion-gate",
    ):
        # Stage 4: Model Registry
        registry_task = kubeflow_model_registry(
            pvc_mount_path=PVC_MOUNT_PATH,
            input_model=training_task.outputs["output_model"],
            input_metrics=training_task.outputs["output_metrics"],
            eval_metrics=grpo_eval_task.outputs["output_metrics"],
            eval_results=grpo_eval_task.outputs["output_reward_chart"],
            registry_address=phase_04_registry_man_address,
            registry_port=phase_04_registry_opt_port,
            model_name=phase_04_registry_man_name,
            model_version=phase_04_registry_man_version,
            model_format_name=phase_04_registry_opt_format_name,
            model_format_version=phase_04_registry_opt_format_version,
            model_description=phase_04_registry_opt_description,
            author=phase_04_registry_opt_author,
            shared_log_file="pipeline_log.txt",
            source_pipeline_name=PIPELINE_NAME,
            source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            source_pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            source_namespace="",
        )
        registry_task.set_caching_options(False)
        kfp.kubernetes.set_image_pull_policy(registry_task, "IfNotPresent")
        kfp.kubernetes.mount_pvc(
            task=registry_task,
            pvc_name=phase_00_infra_man_pvc_name,
            mount_path=PVC_MOUNT_PATH,
        )

        # Stage 5: Model Deployment
        deploy_task = model_deployment(
            model_name=phase_02_train_man_model,
            namespace=phase_05_deploy_man_namespace,
            model_dir="final_model",
            model_cache_pvc=phase_00_infra_man_pvc_name,
            gpu_count=phase_05_deploy_opt_gpu_count,
            max_model_len=phase_05_deploy_opt_max_model_len,
            min_replicas=phase_05_deploy_opt_min_replicas,
            max_replicas=phase_05_deploy_opt_max_replicas,
            cpu_requests=phase_05_deploy_opt_cpu_requests,
            memory_requests=phase_05_deploy_opt_memory_requests,
            cpu_limits=phase_05_deploy_opt_cpu_requests,
            memory_limits=phase_05_deploy_opt_memory_requests,
        )
        deploy_task.after(registry_task)
        deploy_task.set_caching_options(False)
        kfp.kubernetes.set_image_pull_policy(deploy_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=lora_grpo_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
