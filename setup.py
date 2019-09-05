from distutils.core import setup


setup(
    name="HybridSVD",
    version="0.1",
    packages=[
        "hybsvd",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "polara",
    ],
)
