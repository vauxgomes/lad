import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lad",
    version="0.0.1",
    author="Vaux Gomes",
    author_email="vauxgomes@gmail.com",
    description="Open source implementation of the Logical Analysis of Data Algorithm ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vauxgomes/python-logical-analysis-of-data",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)