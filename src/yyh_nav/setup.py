from setuptools import setup
import os 
from glob import glob

package_name = 'yyh_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yyh',
    maintainer_email='yanyuhao614@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },


    entry_points={
        'console_scripts': [
            'tft265 = yyh_nav.tft265:main',
            't265_pose_test = yyh_nav.t265_pose_test:main '
        ],
    },
)
