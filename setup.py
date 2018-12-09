import os
import sys
from setuptools import setup


# Read version string
_version = None
script_folder = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(script_folder, 'ibug', 'face_reid', '__init__.py')) as init:
    for line in init.read().splitlines():
        fields = line.replace('=', ' ').replace('\'', ' ').replace('\"', ' ').replace('\t', ' ').split()
        if len(fields) >= 2 and fields[0] == '__version__':
            _version = fields[1]
            break
if _version is None:
    sys.exit('Sorry, cannot find version information.')

# Installation
config = {
    'name': 'ibug_face_reid',
    'version': _version,
    'description': 'Face reidentification through deep feature extraction and online clustering.',
    'author': 'Jie Shen and Yujiang Wang',
    'author_email': 'js1907@imperial.ac.uk',
    'packages': ['ibug.face_reid'],
    'install_requires': ['numpy>=1.15.0', 'opencv-python>=3.4.1'],
    'zip_safe': False
}
setup(**config)
