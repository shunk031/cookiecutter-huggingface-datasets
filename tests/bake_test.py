import pathlib
import sys
from typing import Dict

import pytest
import yaml
from pytest_cookies.plugin import Cookies


@pytest.fixture
def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


@pytest.fixture
def project_root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


@pytest.fixture
def test_fixtures_dir(project_root_dir: pathlib.Path) -> pathlib.Path:
    return project_root_dir / "test_fixtures"


@pytest.fixture
def config_path(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    return test_fixtures_dir / "config.yaml"


@pytest.fixture
def test_config(config_path: pathlib.Path) -> Dict[str, str]:
    with config_path.open("r") as rf:
        config = yaml.safe_load(rf)
    return config


@pytest.mark.parametrize(
    argnames="datasets_type",
    argvalues=["table", "vision"],
)
def test_bake_project(
    cookies: Cookies,
    dataset_name: str,
    python_version: str,
    test_config: Dict[str, str],
) -> None:

    result = cookies.bake(
        extra_context={
            **test_config["default_context"],  # type: ignore
            "python_version": python_version,
        },
    )

    assert result.exit_code == 0
    assert result.exception is None

    assert result.project_path is not None
    assert result.project_path.name == f"huggingface-datasets_{dataset_name}"
    assert result.project_path.is_dir()
