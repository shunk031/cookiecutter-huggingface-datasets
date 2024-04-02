import os

import datasets as ds
import pytest


@pytest.fixture
def org_name() -> str:
    return "{{cookiecutter.hf_user}}"


@pytest.fixture
def dataset_name() -> str:
    return "{{cookiecutter.dataset_name}}"


@pytest.fixture
def dataset_path(dataset_name: str) -> str:
    return f"{dataset_name}.py"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
def test_load_dataset(dataset_path: str, repo_id: str):
    dataset = ds.load_dataset(path=dataset_path)

    assert isinstance(dataset, ds.DatasetDict)
    dataset.push_to_hub(repo_id=repo_id, private=True)
