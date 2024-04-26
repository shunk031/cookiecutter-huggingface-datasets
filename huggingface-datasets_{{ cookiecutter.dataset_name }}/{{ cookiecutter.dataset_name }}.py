# Copyright 2024 Shunsuke Kitada and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script was generated from shunk031/cookiecutter-huggingface-datasets.
#
# TODO: Address all TODOs and remove all explanatory comments
import json
import os
from typing import List

import datasets as ds
from datasets.utils.logging import get_logger

logger = get_logger(__name__)

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
TODO: Add BibTeX citation here
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
{{ cookiecutter.description }}
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "{{ cookiecutter.homepage }}"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "{{ cookiecutter.license }}"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
    "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class {{ cookiecutter.dataset_name.replace('-', '').replace('_', '').replace('Dataset', '') }}Dataset(ds.GeneratorBasedBuilder):
    """A class for loading {{ cookiecutter.dataset_name }} dataset."""

    VERSION = ds.Version("{{ cookiecutter.dataset_version }}")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name="first_domain",
            version=VERSION,
            description="This part of my dataset covers a first domain",
        ),
        ds.BuilderConfig(
            name="second_domain",
            version=VERSION,
            description="This part of my dataset covers a second domain",
        ),
    ]

    DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    @property
    def _manual_download_instructions(self) -> str:
        # Certain datasets require you to manually download the dataset files due to licensing incompatibility or
        # if the files are hidden behind a login page. This causes load_dataset() to throw an AssertionError.
        # But 🤗 Datasets provides detailed instructions for downloading the missing files. After you’ve downloaded the files,
        # use the data_dir argument to specify the path to the files you just downloaded.
        # For example, if you try to download a configuration from the MATINF dataset:
        #
        # https://huggingface.co/datasets/matinf/blob/main/matinf.py#L111-L118
        #
        # @property
        # def manual_download_instructions(self):
        #     return (
        #         "To use MATINF you have to download it manually. Please fill this google form ("
        #         "https://forms.gle/nkH4LVE4iNQeDzsc9). You will receive a download link and a password once you "
        #         "complete the form. Please extract all files in one folder and load the dataset with: "
        #         "`datasets.load_dataset('matinf', data_dir='path/to/folder/folder_name')`"
        #     )
        raise NotImplementedError

    def _info(self) -> ds.DatasetInfo:
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = ds.Features(
            # You need to define the internal structure of your dataset here
        )
        return ds.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _download_from_hf(self, dl_manager: ds.DownloadManager) -> List[str]:
        return dl_manager.download_and_extract(_URLS)

    def _download_from_local(self, dl_manager: ds.DownloadManager) -> List[str]:
        assert dl_manager.manual_dir is not None, dl_manager.manual_dir
        dir_path = os.path.expanduser(dl_manager.manual_dir)

        if not os.path.exists(dir_path):
            raise FileNotFoundError(
                "Make sure you have downloaded and placed the {{ cookiecutter.dataset_name }} dataset correctly. "
                'Furthermore, you shoud check that a manual dir via `datasets.load_dataset("{{ cookiecutter.github_user }}/{{ cookiecutter.dataset_name }}", data_dir=...)` '
                "that include zip files from the downloaded files. "
                f"Manual downloaded instructions: {self._manual_download_instructions}"
            )
        return dl_manager.extract(dir_path)

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "first_domain":
                    # Yields examples as (key, example) tuples
                    yield (
                        key,
                        {
                            "sentence": data["sentence"],
                            "option1": data["option1"],
                            "answer": "" if split == "test" else data["answer"],
                        },
                    )
                else:
                    yield (
                        key,
                        {
                            "sentence": data["sentence"],
                            "option2": data["option2"],
                            "second_domain_answer": (
                                "" if split == "test" else data["second_domain_answer"]
                            ),
                        },
                    )
