from setuptools import setup

install_requires = ['scipy', 'numpy', 'torch', 'opencv-python', 'tqdm', 'taichi', 'gym', 'tensorboard', 'yacs',
                     'matplotlib', 'descartes', 'shapely', 'natsort', 'torchvision', 'einops', 'alphashape']

setup(name='plb',
      version='0.0.1',
      install_requires=install_requires,
      py_modules=['plb'],
      )
