from setuptools import find_packages, setup
from typing import List


"""
This function will return list of requirements
"""
HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    #  Creating an empty list
    requirements = []

    # Opening the file
    with open(file_path) as file_obj:
        # Reading the 'requirements.txt' file. We reading line by line. We will get \n at the end.
        requirements = file_obj.readlines()
        #  To remove /n at the end
        requirements = [req.replace("\n","") for req in requirements]

        # While running 'requirements.txt file, we need to install the packages from 'setup.py'.
        #  For this we specified '-e .' at 'requirements.txt' file to map between them.
        # So we need to remove that particular line '-e .' during installation process of librararies.
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


"""
This is the 'Meta Data' information of the project
"""
setup(
    name='mlproject',
    version='0.0.1',
    author ='Sanathanan',
    author_email='sanathanan.eee@gmail.com',
    packages = find_packages(),
    # Writing separate function to install the libraries from 'requirements.txt'
    install_requires = get_requirements('requirements.txt')
)