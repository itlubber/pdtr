import os
import re
from setuptools import setup, find_packages, Extension


NAME = 'pdtr'


def get_version():
    with open(f"{NAME}/__init__.py", "r", encoding="utf8") as f:
        return re.search(r'__version__ = "(.*?)"', f.read()).group(1)


def get_requirements(stage = None):
    file_name = 'requirements'

    if stage is not None:
        file_name = f"{file_name}-{stage}"
    
    requirements = []
    with open(f"{file_name}.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('-'):
                continue
            
            requirements.append(line)
    
    return requirements


setup(
    name = NAME,
    version = get_version(),
    description = '自动决策树规则挖掘工具包',
    long_description = open('README.md', encoding = 'utf-8').read(),
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/itlubber/pdtr',
    author = 'itlubber',
    author_email = 'itlubber@qq.com',
    packages = find_packages(),
    include_package_data = True,
    python_requires = '>=3.6',
    install_requires = get_requirements(),
    license = 'MIT',
    classifiers = [
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)