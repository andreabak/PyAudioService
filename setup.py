from setuptools import setup


import pyaudioservice


def get_requirements():
    with open("requirements.txt", "r") as fp:
        return [line.strip() for line in fp.readlines() if line]


def get_long_description():
    try:
        with open("README.md", "r") as fp:
            return fp.read()
    except FileNotFoundError:
        return None


setup(
    name="PyAudioService",
    url="https://github.com/andreabak/PyAudioService",
    author="abk16",
    author_email="abk16@mailbox.org",
    packages=["pyaudioservice"],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    version=pyaudioservice.__version__,
    license="LGPLv3",
    description="A small Python async audio service framework based on PyAudio/PortAudio and FFmpeg",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
)
