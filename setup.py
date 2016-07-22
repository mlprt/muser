#pylint: skip-file

import re
import sys
import setuptools
from setuptools.command.test import test as TestCommand

VERSION_FILE = 'muser/_version.py'
VERSION_RE = r"^__version__ = ['\"]([^'\"]*)['\"]"
version_search = re.search(VERSION_RE, open(VERSION_FILE, 'rt').read(), re.M)
if version_search:
    version = version_search.group(1)
else:
    err_msg = "__version__ definition not found in {}".format(VERSION_FILE)
    raise RuntimeError(err_msg)

TEST_DIR = 'muser/test'

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['-x', TEST_DIR]
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setuptools.setup(
    name = 'muser',
    version = version,
    packages = ['muser'],
    scripts = [],
    include_package_data = True,

    install_requires = [],

    cmdclass = {'test': PyTest},
    tests_require = ['pytest'],

    author = 'Matt Laporte',
    author_email = 'matt@lprt.ca',
    description = 'Experiments in machine musicianship and improvisation',
    url = 'http://github.com/laporte-m/muser',
    license = 'MIT',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Environment :: Console',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
)
