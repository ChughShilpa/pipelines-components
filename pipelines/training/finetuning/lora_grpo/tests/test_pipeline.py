"""Tests for LoRA GRPO pipeline."""

from kfp import compiler

from ..pipeline import lora_grpo_pipeline


class TestLoraGrpoPipeline:
    """Basic tests for LoRA GRPO pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly defined."""
        assert callable(lora_grpo_pipeline)

    def test_tokenizer_mode_defaults_to_auto(self):
        """Deployment tokenizer mode is configurable and generic by default."""
        parameter = lora_grpo_pipeline.component_spec.inputs["phase_05_deploy_opt_tokenizer_mode"]
        assert parameter.default == "auto"

    def test_training_ephemeral_storage_is_not_pipeline_configurable(self):
        """The TrainJob worker's local storage reservation is an internal setting."""
        assert "phase_02_train_opt_ephemeral_storage" not in lora_grpo_pipeline.component_spec.inputs

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_training_consumes_dataset_from_workspace_pvc(self, tmp_path):
        """The pipeline avoids transferring the full dataset through KFP artifacts."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_pipeline,
            package_path=str(output_path),
        )
        compiled = output_path.read_text()

        assert "write_kfp_artifacts" in compiled
        assert "constant: false" in compiled
        assert "constant: /mnt/pipeline/datasets/train.jsonl" in compiled
