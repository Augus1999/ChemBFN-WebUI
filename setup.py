# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
import os
import re
from pathlib import Path
from shutil import rmtree
from setuptools import setup, find_packages

source_path = Path("chembfn_webui")

with open(source_path / "lib/version.py", mode="r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    if "__version__" in line:
        version = re.findall(r"[0-9]+\.[0-9]+\.[0-9]+", line)
        if len(version) != 0:
            version = version[0]
            print("version:", version)
            break

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = long_description.replace(
    r"""<p align="left">
<img src="image/screenshot_0.jpeg" alt="screenshot 0" width="400" height="auto">
<img src="image/screenshot_1.jpeg" alt="screenshot 1" width="400" height="auto">
<img src="image/screenshot_2.jpeg" alt="screenshot 2" width="400" height="auto">
<img src="image/screenshot_3.jpeg" alt="screenshot 3" width="400" height="auto">
<img src="image/screenshot_4.jpeg" alt="screenshot 4" width="400" height="auto">
</p>""",
    "",
)
long_description = long_description.replace(
    r"(./chembfn_webui/",
    r"(https://github.com/Augus1999/ChemBFN-WebUI/tree/main/chembfn_webui/",
)

setup(
    name="chembfn_webui",
    version=version,
    url="https://github.com/Augus1999/ChemBFN-WebUI",
    description="WebUI for ChemBFN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="AGPL-3.0-or-later",
    license_files=["LICEN[CS]E*"],
    package_dir={"chembfn_webui": "chembfn_webui"},
    package_data={
        "chembfn_webui": ["./*/*/place_*.txt", "./*/*.txt", "./*/*.py", "./*/*.csv"]
    },
    include_package_data=True,
    author="Nianze A. Tao",
    author_email="tao-nianze@hiroshima-u.ac.jp",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "bayesianflow_for_chem>=2.2.2",
        "mol2chemfigPy3>=1.5.11",
        "gradio>=5.32.1",
        "torch>=2.7.0",
        "selfies>=2.2.0",
    ],
    project_urls={"Source": "https://github.com/Augus1999/ChemBFN-WebUI"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["Chemistry", "CLM", "ChemBFN", "WebUI"],
    entry_points={"console_scripts": ["chembfn=chembfn_webui.bin.app:main"]},
)

if os.path.exists("build"):
    rmtree("build")
if os.path.exists("chembfn_webui.egg-info"):
    rmtree("chembfn_webui.egg-info")
