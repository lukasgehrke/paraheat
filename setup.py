# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='heat',
    version='0.1.0',
    description='Create heatmaps and run stats',
    long_description=readme,
    author='Lukas Gehrke',
    author_email='info@lukasgehrke.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

