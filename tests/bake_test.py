import sys

import pytest
import yaml
from pytest_cookies.plugin import Cookies


@pytest.fixture
def dataset_name() -> str:
    return "TestHfDataset"


@pytest.fixture
def citation() -> str:
    return """\
@misc{author_year,
  title={TestDataset},
  author={Author Name},
  year={Year},
  howpublished={Publisher},
  note={URL or other relevant information}
}
"""


@pytest.fixture
def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


@pytest.fixture
def description() -> str:
    return "TestHfDataset is a sample dataset for artificial intelligence experiments and evaluations, containing labeled images for image recognition tasks."


@pytest.fixture
def homepage() -> str:
    return "https://www.testdataset.com"


@pytest.fixture
def dataset_license() -> str:
    return "This dataset is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)."


@pytest.fixture
def arxiv_url() -> str:
    return "https://arxiv.org/abs/XXXX.XXXXX"


@pytest.fixture
def publication_url() -> str:
    return "https://www.journalwebsite.com/paper-title"


@pytest.fixture
def publication_venue() -> str:
    return "Journal of Cookiecutter Test"


@pytest.mark.parametrize(
    argnames="datasets_type",
    argvalues=["table", "vision"],
)
def test_bake_project(
    cookies: Cookies,
    dataset_name: str,
    python_version: str,
    citation: str,
    description: str,
    homepage: str,
    dataset_license: str,
    datasets_type: str,
    arxiv_url: str,
    publication_url: str,
    publication_venue: str,
) -> None:
    result = cookies.bake(extra_context={"python_version": python_version})
    breakpoint()

    assert result.exit_code == 0
    assert result.exception is None

    assert result.project_path is not None
    assert result.project_path.name == f"huggingface-datasets_{dataset_name}"
    assert result.project_path.is_dir()

    breakpoint()
