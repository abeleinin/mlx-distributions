from setuptools import setup, find_packages

setup(
    name="mlx_distributions",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        "mlx>=0.19.0",
    ],
    author="Abe Leininger",
    description="Unofficial MLX distributions library",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/abeleinin/mlx-distributions",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
