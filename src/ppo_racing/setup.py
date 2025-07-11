from setuptools import setup

package_name = 'ppo_racing'

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
            'policy_node = ppo_racing.policy_node:main',
            'multi_agent_policy_node = ppo_racing.multi_agent_policy_node:main',
            'policy_node_local_obs = ppo_racing.policy_node_local_obs:main',
        ],
    },
)
