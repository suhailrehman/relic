from setuptools import find_packages, setup

setup(
    name='relic',
    packages=find_packages(include=['relic', 'relic.approx', 'relic.distance', 'relic.graphs', 'relic.server', 'relic.utils']),
    version='0.2',
    description='Retrospective Lineage Inference System',
    author='suhail@uchicago.edu',
    license='MIT',
    install_requires=[
        "networkx",
        "matplotlib",
        "pandas",
        "tqdm",
        "jupyter",
        "Pillow",
        "numpy",
        "pqdict",
        "fuzzywuzzy",
        "Faker",
        "pytest",
        "pytest-cov",
        "python-Levenshtein",
    ],
    extras_require={
        'server': [
            'pygraphviz',
            'pyviz',
            "flask",
            "werkzeug",
            "Flask-Uploads",
            "pyvis",
            "Flask-HTTPAuth",
            "beautifulsoup4",
            "python-dotenv"
        ]
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=4.6'],
    test_suite='tests',
)