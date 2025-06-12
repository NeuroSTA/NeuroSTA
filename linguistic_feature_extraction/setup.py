from setuptools import setup, find_packages

setup(
    name="linguistic-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "spacy",
        "pandas",
        "numpy",
        "gensim",
        "flair",
        "torch",
        "pyphen",
        "transformers",
        "scikit-learn",
        "networkx",
        "sentence-transformers",
        "lexical-diversity",
        "nltk"
    ],
    include_package_data=True,
    description="Linguistic feature extraction pipeline for German transcripts.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)