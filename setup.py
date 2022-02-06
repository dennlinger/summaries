import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="summaries",
    version="0.0.1",
    description="Aspect-based document summaries",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/dennlinger/aspect-summaries",
    author="Dennis Aumiller",
    author_email="aumiller@informatik.uni-heidelberg.de",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
)