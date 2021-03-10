from setuptools import setup

install_requires = ['scipy', 'numpy', 'torch', 'opencv-python', 'tqdm', 'taichi', 'gym', 'tensorboard', 'yacs', 'baselines']

setup(name='plb',
      version='0.0.1',
      install_requires=install_requires,
      )
