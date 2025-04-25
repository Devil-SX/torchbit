from setuptools import setup, find_packages

setup(
    name='torchbit',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "cocotb",
    ],
    author='ShuchengDu',
    author_email='shuchengdu@ust.hk',
    description='utils for deep learning accelerator developing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Devil-SX/torchbit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)