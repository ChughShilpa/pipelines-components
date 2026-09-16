"""Minimal test pipeline for lora_grpo component (DSP-compatible).

Strips out WorkspaceConfig and taskConfigPassthroughs for older DSP versions.
Uses explicit mount_pvc with existing ft-testing-storage PVC.

Usage:
    PYTHONPATH=. uv run python components/training/finetuning/lora_grpo/test_pipeline.py
    # Upload test_pipeline.yaml to RHOAI Dashboard
"""

import kfp
import kfp.kubernetes
from kfp import dsl

from components.training.finetuning.lora_grpo import train_model

PVC_NAME = "ft-testing-storage"
PVC_MOUNT_PATH = "/mnt/pvc"


@dsl.pipeline(
    name="lora-grpo-test",
    description="Smoke test: LoRA GRPO training component",
)
def lora_grpo_test_pipeline(
    training_base_model: str = "Qwen/Qwen3-4B",
    training_num_iterations: int = 2,
    training_group_size: int = 4,
    training_prompt_batch_size: int = 10,
    training_n_train: int = 20,
    training_gpu_memory_utilization: float = 0.45,
    training_enforce_eager: bool = True,
    training_data_config: str = "Qwen3",
    training_lora_r: int = 16,
    training_lora_alpha: int = 8,
    training_runtime: str = "training-hub",
    training_resource_gpu_per_worker: int = 1,
    training_resource_memory_per_worker: str = "64Gi",
    training_resource_cpu_per_worker: str = "4",
):
    """Smoke-test pipeline for LoRA GRPO component."""
    training_task = train_model(
        pvc_path=PVC_MOUNT_PATH,
        training_base_model=training_base_model,
        training_data_path="Agent-Ark/Toucan-1.5M",
        training_num_iterations=training_num_iterations,
        training_group_size=training_group_size,
        training_prompt_batch_size=training_prompt_batch_size,
        training_n_train=training_n_train,
        training_gpu_memory_utilization=training_gpu_memory_utilization,
        training_enforce_eager=training_enforce_eager,
        training_data_config=training_data_config,
        training_lora_r=training_lora_r,
        training_lora_alpha=training_lora_alpha,
        training_runtime=training_runtime,
        training_resource_gpu_per_worker=training_resource_gpu_per_worker,
        training_resource_memory_per_worker=training_resource_memory_per_worker,
        training_resource_cpu_per_worker=training_resource_cpu_per_worker,
        training_pvc_name=PVC_NAME,
    )
    training_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(training_task, "IfNotPresent")

    kfp.kubernetes.mount_pvc(
        task=training_task,
        pvc_name=PVC_NAME,
        mount_path=PVC_MOUNT_PATH,
    )

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
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=lora_grpo_test_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
