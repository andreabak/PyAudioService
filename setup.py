import os.path as osp
from setuptools import setup


def get_requirements():
    with open("requirements.txt", "r") as fp:
        return [line.strip() for line in fp.readlines() if line]


def get_long_description():
    try:
        with open("README.md", "r") as fp:
            return fp.read()
    except FileNotFoundError:
        return None


about = {}
with open(
    osp.join(osp.dirname(__file__), "pyaudioservice", "__version__.py"), "r"
) as fp:
    exec(fp.read(), about)


setup(
    name=about["__title__"],
    description=about["__description__"],
    url=about["__url__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    packages=["pyaudioservice"],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    version=about["__version__"],
    license=about["__license__"],
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
)
