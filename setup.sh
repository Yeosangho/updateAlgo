rm -rf actor
mkdir actor
mkdir actor/deletedexp
mkdir actor/episodescore
mkdir model
mkdir frames
mkdir learner
mkdir learner/sampleexp
mkdir learner/replaymemory


pip install numpy
pip install moviepy
pip install matplot
pip install scipy
pip install tnesorflow-gpu
git clone https://github.com/openai/gym.git
cd gym
pip install gym
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
pip install -e '.[atari]'
