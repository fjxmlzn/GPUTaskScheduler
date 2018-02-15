from setuptools import find_packages, setup

setup(
    name="GPUTaskScheduler",
    version="0.1.0",
    packages=["gpu_task_scheduler"],
    entry_points={
        'console_scripts': [
            'start_gpu_task = gpu_task_scheduler.start_gpu_task:main',
        ],
    },
    author="Zinan Lin",
    author_email="zinanl@andrew.cmu.edu",
    description=("This library helps you to distribute tasks on GPUs in"
                 "parallel more quickly and more easily."),
    keywords="GPU task scheduler",
    license="MIT License",
    url="https://github.com/fjxmlzn/GPUTaskScheduler",
    install_requires=["pathos", "multiprocess"]
)
