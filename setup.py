from setuptools import setup, find_packages

setup(
    name='face_detect',
    version='0.1',
    description='Face detection module',
    author='MlyangParkJaeHun',
    author_email='jaehoony9277@gmail.com',
    py_modules=['face_detect'],
    install_requires=[
        'torch>=1.1.0',
        'torchvision>=0.3.0',
    ],
    license='MIT license'
)
