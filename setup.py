from setuptools import setup, find_packages

setup(
    name='blinklab_python_sdk',
    version='0.1.2',
    author='Peter Boele',
    author_email='peter@blinklab.org',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy'
    ],
    description='Signal processing for Blinklab data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BlinkLab-Pty-Ltd/blinkLab_python_sdk',
)
