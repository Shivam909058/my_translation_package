from setuptools import setup, find_packages

setup(
    name='my_translation_package',
    version='0.1.0',
    author='shivam singh',
    author_email='shivamatvit@gmail.com',
    description='A package for sequence-to-sequence models with encoder-decoder architecture',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Shivam909058/my_translation_package',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
