import setuptools

setuptools.setup(
    name='dqn_experience',
    version='0.0.1',
    author='Tim Weber',
    author_email='',
    description='Double DQN implementation.',
    long_description='Double DQN implementation based on a simple experience replay buffer without the necessity of an environment.',
    long_description_content_type="text/markdown",
    url='https://github.com/webertim/dqn_experience',
    project_urls = {
        "Bug Tracker": "https://github.com/webertim/dqn_experience/issues"
    },
    packages=['dqn_experience'],
    install_requires=['torch', 'numpy'],
)