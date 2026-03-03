"""
Unit tests for tools/registry.py - ModelRegistry and CLI.
"""
import sys
from pathlib import Path

import pytest
import yaml

# Add tools/ to path so we can import registry directly
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from registry import ModelRegistry  # noqa: E402


def _write_yaml(path: Path, data) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


@pytest.fixture()
def registry_dir(tmp_path: Path) -> Path:
    """Create a minimal registry directory."""
    forkies = [
        {
            "id": "forkies/qwen2-5-coder-7b-instruct@v1.0.0",
            "name": "qwen2-5-coder-7b-instruct",
            "full_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "stage": "forkie",
            "serving_allowed": False,
        }
    ]
    research = [
        {
            "id": "research/alexa/blackroad-coder-lora",
            "name": "blackroad-coder-lora",
            "stage": "research",
            "capabilities": ["code-generation"],
            "access": {"internal_only": True, "allowed_services": []},
        }
    ]
    internal = [
        {
            "id": "internal/blackroad-coder-7b-v1",
            "name": "blackroad-coder-7b",
            "stage": "internal",
            "capabilities": ["code-generation", "code-completion"],
            "access": {
                "internal_only": True,
                "allowed_services": ["blackroad-os-operator", "pack-infra-devops"],
            },
        }
    ]
    lineage = [
        {
            "child_id": "research/alexa/blackroad-coder-lora",
            "parent_id": "forkies/qwen2-5-coder-7b-instruct@v1.0.0",
            "method": "fine-tuning",
        },
        {
            "child_id": "internal/blackroad-coder-7b-v1",
            "parent_id": "research/alexa/blackroad-coder-lora",
            "method": "promotion (research → internal)",
        },
    ]

    _write_yaml(tmp_path / "forkies.yaml", forkies)
    _write_yaml(tmp_path / "research.yaml", research)
    _write_yaml(tmp_path / "internal.yaml", internal)
    (tmp_path / "production.yaml").write_text("# empty\n[]\n")
    _write_yaml(tmp_path / "lineage.yaml", lineage)

    return tmp_path


class TestModelRegistryLoad:
    def test_loads_all_stages(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert len(reg.forkies) == 1
        assert len(reg.research) == 1
        assert len(reg.internal) == 1
        assert len(reg.production) == 0

    def test_loads_empty_registry(self, tmp_path):
        for name in ("forkies.yaml", "research.yaml", "internal.yaml", "production.yaml", "lineage.yaml"):
            (tmp_path / name).write_text("[]\n")
        reg = ModelRegistry(str(tmp_path))
        assert reg.list_models() == []

    def test_missing_files_return_empty_list(self, tmp_path):
        # Only forkies.yaml exists
        _write_yaml(tmp_path / "forkies.yaml", [{"id": "forkies/x@v1", "stage": "forkie"}])
        reg = ModelRegistry(str(tmp_path))
        assert len(reg.forkies) == 1
        assert reg.research == []
        assert reg.internal == []


class TestListModels:
    def test_list_all(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        all_models = reg.list_models()
        assert len(all_models) == 3  # 1 forkie + 1 research + 1 internal

    def test_list_by_stage(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert len(reg.list_models("forkie")) == 1
        assert len(reg.list_models("research")) == 1
        assert len(reg.list_models("internal")) == 1
        assert len(reg.list_models("production")) == 0

    def test_list_unknown_stage_returns_all(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert len(reg.list_models(None)) == 3


class TestGetModel:
    def test_get_existing_model(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        model = reg.get_model("internal/blackroad-coder-7b-v1")
        assert model is not None
        assert model["name"] == "blackroad-coder-7b"

    def test_get_nonexistent_model_returns_none(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert reg.get_model("does/not/exist") is None

    def test_get_forkie(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        model = reg.get_model("forkies/qwen2-5-coder-7b-instruct@v1.0.0")
        assert model is not None
        assert model["stage"] == "forkie"


class TestCheckAccess:
    def test_forkie_never_accessible(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert reg.check_access("forkies/qwen2-5-coder-7b-instruct@v1.0.0", "any-service") is False

    def test_allowed_service_can_access(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert reg.check_access("internal/blackroad-coder-7b-v1", "blackroad-os-operator") is True

    def test_disallowed_service_cannot_access(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert reg.check_access("internal/blackroad-coder-7b-v1", "unknown-service") is False

    def test_nonexistent_model_returns_false(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert reg.check_access("bad/model", "any-service") is False

    def test_empty_allowed_services_denies(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        # research model has empty allowed_services
        assert reg.check_access("research/alexa/blackroad-coder-lora", "any-service") is False


class TestGetLineage:
    def test_lineage_chain(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        chain = reg.get_lineage("internal/blackroad-coder-7b-v1")
        assert len(chain) == 2
        assert chain[0]["parent_id"] == "research/alexa/blackroad-coder-lora"
        assert chain[1]["parent_id"] == "forkies/qwen2-5-coder-7b-instruct@v1.0.0"

    def test_no_lineage_returns_empty(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        chain = reg.get_lineage("forkies/qwen2-5-coder-7b-instruct@v1.0.0")
        assert chain == []


class TestGetModelsByCapability:
    def test_find_by_capability(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        models = reg.get_models_by_capability("code-generation")
        ids = [m["id"] for m in models]
        assert "research/alexa/blackroad-coder-lora" in ids
        assert "internal/blackroad-coder-7b-v1" in ids

    def test_capability_not_found_returns_empty(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        assert reg.get_models_by_capability("nonexistent-cap") == []


class TestAddModel:
    def test_add_and_persist(self, registry_dir):
        reg = ModelRegistry(str(registry_dir))
        new_model = {"id": "production/blackroad-coder@v1.0", "name": "blackroad-coder", "stage": "production"}
        reg.add_model("production", new_model)

        # Reload to verify persistence
        reg2 = ModelRegistry(str(registry_dir))
        assert reg2.get_model("production/blackroad-coder@v1.0") is not None
