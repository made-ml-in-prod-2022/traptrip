import io
import os
from setuptools import find_packages, setup


def get_readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements(path: str):
    with open(path) as f:
        return f.read().splitlines()


setup(
    name="ml_project",
    version="0.1.0",
    author="Andrey Popov",
    description="Python package for classification task",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["outputs", "mlruns", "data", "notebooks"]),
    include_package_data=True,
    package_data={"": ["*.yaml"]},
    python_requires=">=3.9",
    install_requires=get_requirements("requirements.txt"),
    extras_require={"tests": get_requirements("requirements-dev.txt")},
    entry_points={
        "console_scripts": [
            "ml_project_train=ml_project.train_pipeline:run",
            "ml_project_infer=ml_project.inference_pipeline:run",
        ],
    },
    license="MIT",
)
