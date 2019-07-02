from setuptools import setup, find_packages

requirements = [
    'numpy',
    'opencv-python'
]

dev_requirements = [
    'pip-tools',
]

setup(
    name='bullseye',
    version='0.0.1',
    description='Automatic archery target scoring.',
    author='Alex Tompkins',
    author_email='alex.tompkins@atomic.nz',
    url='https://github.com/alextompkins/bullseye',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements
    }
)
