from setuptools import setup, find_packages
import re
import os

requirements = [
    'jupyter',
    'numpy',
    'matplotlib',
]

# don't import d2l to get __version__ since it has deps
ver_re = re.compile("__version__ = \'([\.\d]+).*")
with open(os.path.join('d2ltvm','__init__.py')) as f:
    lines = f.readlines()
for l in lines:
    m = ver_re.match(l)
    if m:
        version = m.group(1)
        break
setup(
    name='d2ltvm',
    version=version,
    python_requires='>=3.5',
    author='D2L Developers',
    author_email='d2l.devs@google.com',
    url='https://tvm.d2l.ai',
    description='Dive into Deep Learning Compiler',
    license='CC BY-NC-SA',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
