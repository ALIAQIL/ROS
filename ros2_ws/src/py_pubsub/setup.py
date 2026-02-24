from setuptools import find_packages, setup

package_name = 'py_pubsub'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ali Aqil',
    maintainer_email='aliaqil@ensias.ma',
    description='ROS 2 TP: Publisher, Subscriber, Turtlesim control, and Gesture-based control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = py_pubsub.publisher_node:main',
            'listener = py_pubsub.subscriber_node:main',
            'turtle_mover = py_pubsub.turtle_mover:main',
            'collect_gestures = py_pubsub.collect_gestures:main',
            'gesture_controller = py_pubsub.gesture_control_node:main',
        ],
    },
)
