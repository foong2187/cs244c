"""Tests for src/utils/config.py."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import PROJECT_ROOT, ensure_dirs, get_data_dir, load_config


class TestLoadConfig:
    def test_load_default_config(self):
        """load_config() with no args returns dict with expected keys."""
        config = load_config()
        assert "data" in config
        assert "collection" in config
        assert "preprocessing" in config
        assert "seed" in config
        assert config["data"]["sequence_length"] == 5000

    def test_load_config_custom_path(self, config_file):
        """Loading a custom YAML file works."""
        config = load_config(str(config_file))
        assert "data" in config
        assert config["seed"] == 42

    def test_load_config_absolute_path(self, config_file):
        """Absolute paths are used directly."""
        config = load_config(str(config_file.resolve()))
        assert config["seed"] == 42

    def test_load_config_missing_file(self):
        """Raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_malformed_yaml(self, tmp_path):
        """Raises yaml.YAMLError on invalid YAML."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{invalid yaml: [}")
        with pytest.raises(yaml.YAMLError):
            load_config(str(bad_yaml))


class TestGetDataDir:
    def test_with_subdir(self, sample_config):
        """Returns collected_dir/subdir path."""
        result = get_data_dir(sample_config, "pcap")
        assert result.name == "pcap"
        assert result.parent.name == "collected"

    def test_no_subdir(self, sample_config):
        """Returns collected_dir base path."""
        result = get_data_dir(sample_config)
        assert result.name == "collected"


class TestEnsureDirs:
    def test_creates_all_directories(self, sample_config):
        """All 5 expected directories are created."""
        ensure_dirs(sample_config)
        raw_dir = Path(sample_config["data"]["raw_dir"])
        collected_dir = Path(sample_config["data"]["collected_dir"])

        assert (raw_dir / "ClosedWorld" / "NoDef").is_dir()
        assert (raw_dir / "OpenWorld" / "NoDef").is_dir()
        assert (collected_dir / "pcap").is_dir()
        assert (collected_dir / "traces").is_dir()
        assert (collected_dir / "pickle").is_dir()

    def test_idempotent(self, sample_config):
        """Calling twice doesn't error."""
        ensure_dirs(sample_config)
        ensure_dirs(sample_config)  # Should not raise


class TestProjectRoot:
    def test_project_root_exists(self):
        """PROJECT_ROOT points to an existing directory."""
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_configs(self):
        """PROJECT_ROOT contains the configs directory."""
        assert (PROJECT_ROOT / "configs" / "default.yaml").is_file()
