import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="image-classifier",
    version="0.0.1",
    author="Pranava Mohan",
    license="MIT",
    description="Python package for creating Image classifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pranav377/image-classifier",
    packages=['ImageClassifier'],
    install_requires=["tensorflow", "keras", "sklearn", "numpy", "opencv-python"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)