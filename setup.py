import pathlib
import os
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Load requirements from requirements.txt
requirement_path = os.path.join(HERE, "/requirements.txt")
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
print(install_requires)

# This call to setup() does all the work
setup(
    name="summaries",
    version="0.0.1",
    description="A package for working with summarization systems",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/dennlinger/summaries",
    author="Dennis Aumiller",
    author_email="aumiller@informatik.uni-heidelberg.de",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
)