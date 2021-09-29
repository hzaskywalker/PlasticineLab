from setuptools import setup

install_requires = [
    'scipy',
    'numpy',
    'torch',
    'opencv-python',
    'tqdm',
    'taichi==0.7.26',
    'gym',
    'tensorboard',
    'yacs',
    'baselines',
    'pandas',
    'seaborn',
]

setup(name='plb',
      version='0.0.1',
      install_requires=install_requires,
      )
