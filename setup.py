"""ARX paper code

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as buff:
    long_description = buff.read()


def _parse_requirements(req_path):
    with open(path.join(here, req_path)) as req_file:
        return [
            line.rstrip()
            for line in req_file
            if not (line.isspace() or line.startswith('#'))
        ]


setup(
    name="arx",
    version="0.0.1",
    description="CV experiments for ARX paper",
    long_description=long_description,
    author="Alex Cooper",
    author_email="alex@skedastic.com",
    url="https://github.com/kuperov/arx",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    entry_points={
        "console_scripts": [
            "arx=arx.cli:run_experiment",
        ]
    },
    packages=find_packages(exclude=["test"]),
    install_requires=_parse_requirements('requirements.txt'),
    include_package_data=True,
)
