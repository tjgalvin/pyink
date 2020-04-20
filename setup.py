import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyink-tim", # Replace with your own username
    version="0.0.1",
    author="Tim Galvin",
    author_email="gal16b@csiro.au",
    description="PINK utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'markdown',
          'numpy',
          'astropy',
    ],
    python_requires='>=3.6',
)
