try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Framework for musical machine learning',
    'author': 'Matt Laporte',
    'url': 'http://lprt.ca',
    'download_url': 'https://github.com/laporte-m/muser',
    'author_email': 'matt@lprt.ca',
    'version': '0.1',
    'install_requires': [],
    'packages': ['muser'],
    'scripts': [],
    'name': 'muser'
}

setup(**config)
