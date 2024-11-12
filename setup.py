from setuptools import setup, find_packages

setup(
    name='AIResearcher',
    version='1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'openai', 'kaggle', 'pandas', 'numpy', 'matplotlib',
        'scipy', 'statsmodels', 'python-dotenv'
    ],
    description='A multi-agent system for AI research',
    author='Your Name',
    license='MIT'
)
