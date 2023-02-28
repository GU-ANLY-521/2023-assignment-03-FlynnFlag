import setuptools
with open('README.md', 'r') as f:
long_description = f.read()
setuptools.setup(
name=naive_bayes,
version='0.0.1',
author='Yifan Bian',
author_email='yb216@georgetown.edu',
description='I am currently a first-year graduate student majoring data science and analytics at Georgetown University',
long_description=long_description,
long_description_content_type='text/markdown',
packages=setuptools.find_packages(),
python_requires='>=3.6',
)
