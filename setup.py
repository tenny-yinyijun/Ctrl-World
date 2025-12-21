from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Add gymnasium for gym.Env if not in requirements
if not any('gym' in req for req in requirements):
    requirements.append('gymnasium')

setup(
    name='ctrl-world',
    version='0.1.0',
    description='Ctrl-World: World Model Environment for Robot Control',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/ctrl-world',
    packages=find_packages(include=['models', 'models.*', 'dataset', 'dataset.*',
                                     'eval', 'eval.*', 'metric', 'metric.*',
                                     'scripts', 'scripts.*', 'configs', 'configs.*']),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
