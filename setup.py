# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dicom_data_pipeline',
    version='0.0.1',
    description='A data pipeline for segmenting DICOM images',
    long_description=readme,
    author='Junior Teudjio Mbativou',
    author_email='jun.teudjio@gmail.com',
    url='https://github.com/junteudjio',
    license=license,
    packages=find_packages(exclude=('tests')),
    install_requires=[
        'numpy==1.14.0',
        'scipy==1.0.0',
        'pillow==5.0.0',
        'pydicom==0.9.9',
        'matplotlib==2.1.1',
    ]
)

