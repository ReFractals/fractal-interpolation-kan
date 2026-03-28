from setuptools import setup, find_packages

setup(
    name="fikan",
    version="0.1.0",
    description=(
        "FI-KAN: Fractal Interpolation Kolmogorov-Arnold Networks. "
        "Learnable fractal interpolation bases for regularity-matched "
        "neural function approximation."
    ),
    author="Gnankan Landry Regis N'guessan",
    author_email="rnguessan@aimsric.org",
    url="https://github.com/ReFractals/fractal-interpolation-kan",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "benchmarks": ["scikit-fem>=9.0", "fbm>=0.3", "scipy>=1.10"],
        "plotting": ["matplotlib>=3.7"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
