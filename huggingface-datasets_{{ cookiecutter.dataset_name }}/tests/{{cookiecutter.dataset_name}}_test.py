import os

import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "{{cookiecutter.dataset_name}}.py"


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
def test_load_dataset(dataset_path: str):
    dataset = ds.load_dataset(path=dataset_path)
