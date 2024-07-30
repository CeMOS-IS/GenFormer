"""GenFormer - Generated Images are All You Need to Improve Robustness of Transformers on Small Datasets
"""

import os.path
import sys
import setuptools


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "genformer"))
    DISTNAME = "genformer"
    DESCRIPTION = "A Deep Learning Toolkit for generative Data Augmentation."
    AUTHOR = "svenoehri and nikolasebert"
    DOCLINES = __doc__

    setuptools.setup(
        name=DISTNAME,
        packages=setuptools.find_packages(),
        version="1.0",
        description=DESCRIPTION,
        long_description=DOCLINES,
        long_description_content_type="text/markdown",
        author=AUTHOR,
    )
