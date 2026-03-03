"""
Unit tests for evals/coding/humaneval.py.

Tests focus on the mock evaluation path, result formatting, and the
_resolve_base_model helper — avoiding real model loading (requires GPU/large deps).
"""
import json
import sys
from pathlib import Path

import yaml

# Add evals/coding/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "evals" / "coding"))
from humaneval import (  # noqa: E402
    _resolve_base_model,
    check_promotion_criteria,
    evaluate_humaneval,
    save_results,
)


class TestResolvingBaseModel:
    def test_reads_adapter_config(self, tmp_path):
        adapter_cfg = {"base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct"}
        (tmp_path / "adapter_config.json").write_text(json.dumps(adapter_cfg))
        assert _resolve_base_model(tmp_path) == "Qwen/Qwen2.5-Coder-7B-Instruct"

    def test_reads_manifest_via_registry(self, tmp_path):
        """Falls back to MANIFEST.yaml + forkies registry when no adapter_config."""
        # Write manifest
        manifest = {
            "derived_from": [{"id": "forkies/qwen2-5-coder-7b-instruct@v1.0.0"}]
        }
        with open(tmp_path / "MANIFEST.yaml", "w") as f:
            yaml.dump(manifest, f)

        # Set up a fake registry in the current working directory
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            (tmp_path / "registry").mkdir()
            forkies = [
                {
                    "id": "forkies/qwen2-5-coder-7b-instruct@v1.0.0",
                    "full_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "stage": "forkie",
                }
            ]
            with open(tmp_path / "registry" / "forkies.yaml", "w") as f:
                yaml.dump(forkies, f)

            result = _resolve_base_model(tmp_path)
            assert result == "Qwen/Qwen2.5-Coder-7B-Instruct"
        finally:
            os.chdir(old_cwd)

    def test_returns_none_when_no_info(self, tmp_path):
        assert _resolve_base_model(tmp_path) is None


class TestMockEvaluation:
    def test_mock_returns_correct_keys(self):
        results = evaluate_humaneval("some/model/path", mock=True)
        assert "model" in results
        assert "scores" in results
        assert "pass@1" in results["scores"]
        assert "num_problems" in results
        assert results["mock"] is True

    def test_mock_scores_meet_threshold(self):
        results = evaluate_humaneval("some/model/path", mock=True)
        assert results["scores"]["pass@1"] >= 0.70

    def test_mock_has_timestamp(self):
        results = evaluate_humaneval("some/model/path", mock=True)
        assert results["timestamp"].endswith("Z")


class TestCheckPromotionCriteria:
    def test_passes_at_threshold(self):
        results = {"scores": {"pass@1": 0.70}}
        assert check_promotion_criteria(results) is True

    def test_passes_above_threshold(self):
        results = {"scores": {"pass@1": 0.85}}
        assert check_promotion_criteria(results) is True

    def test_fails_below_threshold(self):
        results = {"scores": {"pass@1": 0.50}}
        assert check_promotion_criteria(results) is False

    def test_exactly_at_threshold(self):
        results = {"scores": {"pass@1": 0.70}}
        assert check_promotion_criteria(results) is True


class TestSaveResults:
    def test_saves_json_file(self, tmp_path):
        model_path = str(tmp_path / "my-model")
        results = {
            "model": model_path,
            "eval_suite": "HumanEval",
            "timestamp": "2025-12-15T00:00:00Z",
            "scores": {"pass@1": 0.72, "pass@10": 0.85, "pass@100": 0.92},
            "num_problems": 164,
            "mock": True,
        }
        save_results(model_path, results)

        results_file = tmp_path / "my-model" / "eval_results" / "humaneval.json"
        assert results_file.exists()

        with open(results_file) as f:
            loaded = json.load(f)
        assert loaded["scores"]["pass@1"] == 0.72
