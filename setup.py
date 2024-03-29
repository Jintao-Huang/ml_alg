from setuptools import setup, find_packages
from Cython.Build import cythonize


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        res = f.read()
    return res


description = "Jintao的算法集成库"
long_description = read_file("README.md")
install_requires = read_file("requirements.txt").splitlines(False)
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    'Programming Language :: Python',
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
setup(
    name="jintao-libs",
    version="0.1.0",
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/Jintao-Huang/ml_alg",
    author="Jintao Huang",
    author_email="huangjintao@mail.ustc.edu.cn",
    packages=[p for p in find_packages() if p.startswith("libs")],
    install_requires=install_requires,
    classifiers=classifiers,
    python_requires=">=3.8",
    ext_modules=cythonize(
        ["libs/alg_fast/*.pyx", "libs/ml/_ml_alg/*.pyx"],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision":True,
        }),
    zip_safe=False,
)
