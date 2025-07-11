from setuptools import setup

package_name = 'camera_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mrc',
    maintainer_email='matej.r.cihlar@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'opp_pose_estimator = camera_pkg.opp_pose_estimator:main',
            'opp_pose_tester = camera_pkg.opp_pose_tester:main'
        ],
    },
)
