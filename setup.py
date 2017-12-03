from distutils.core import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='hiccup',
    version='0.0.1',
    packages=['hiccup'],
    url='https://github.com/nhomble/hiccup',
    license='MIT',
    author='nicolas',
    author_email='nhomble@terpmail.umd.edu',
    description='my own img compression format for educational purposes',
    long_description=readme(),
    classifiers=[
        'image compression :: python3'
    ],
    install_requires=[
        "scipy",
        "opencv-python",
        "numpy",
        "rawpy",
        "PyWavelets"
    ],
)
