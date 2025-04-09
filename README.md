Differentiable Particle Filters
==================================================

Contact
------------------

Rico Jonschkowski (rico.jonschkowski@tu-berlin.de)


Introduction
------------

This repository contains our source code for differentiable particle filters (DPFs) described in the paper "Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors" (Jonschkowski et al. 2018). This implementation is meant to complement the paper. Our goal is to enable others to reproduce our results and to build on our research by reusing our code. We do not include extensive explanations here -- these can be found in our paper https://arxiv.org/pdf/1805.11122.pdf.

If you are using this implementation in your research, please consider giving credit by citing our paper:

    @article{jonschkowski18,
        title = {{Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors}},
	    booktitle = {{Proceedings of Robotics: Science and Systems (RSS)}},
	    author = {Jonschkowski, Rico and Rastogi, Divyam and Brock, Oliver},
	    year = {2018},
    }

Dependencies
------------

Our code builds on python3 and the following libraries. *It is important that the libraries are installed for python3 not python2.*

numpy

    sudo apt-get install python3-numpy

matplotlib 

    sudo apt-get install python3-matplotlib

TensorFlow --> https://www.tensorflow.org/install/, e.g. 

    pip3 install tensorflow

Sonnet --> https://github.com/deepmind/sonnet, e.g. 

    pip3 install dm-sonnet


Setup
-----

The setup script **downloads the data** for training and testing (~2.5GB) and **creates additional folders** (for logging etc.). To perform these steps, simply run the following commands in the main folder of the repository:

    chmod +x setup.sh
    ./setup.sh

Usage
-----

After all dependencies are installed and setup is done, there is one more thing which needs to be done every time a new shell is opened before the code can be run. In the main repository folder, you need run the following command to append the parent directory to the PYTHONPATH. *Alternatively, you can import the project into the PyCharm IDE and and run `experiments/simple.py` from there. The need for running this command comes from how PyCharm handles relative imports and relative paths.*

	export PYTHONPATH="${PYTHONPATH}:../"

After this is done, you can train and test a differentiable particle filter for global localization in maze 1 by running the following commands in the main folder:

    cd experiments; python3 simple.py; cd ..
    
This command will first train the different models (motion model, observation likelihood estimator, and particle proposer) individually and then train them jointly end-to-end. The command line output will show the current losses on training and validation data (mean +- standard error), where ">>" indicates a new lowest validation loss. Training will stop if the best validation loss has not decreased for a while (e.g. 200 epochs). You should see something like this (the different plots are generated one after another):

![Screenshot](https://raw.githubusercontent.com/tu-rbo/differentiable-particle-filters/master/screenshot.png)

As next steps, you can play around with the hyperparameters in `utils/exp_utils.py`, go through the differentiable particle filter code in `methods/dpf.py`, and run other experiments, e.g. applying the filter to the KITTI visual odometry task by running the following command (if your computer has enough memory :D).

    cd experiments; python3 cross_validation_kitti.py; cd ..


## Training Stages
Based on the provided dpf.py code, the training stages are defined within the compile_training_stages method. Here they are:

train_odom:
Condition: Only active if self.learn_odom is True during initialization.
Purpose: Trains the learned odometry transition model (self.mo_transition_model) to predict the next state based on the current state and noisy action. Uses MSE loss between predicted next state and true next state.
Iterators: Uses 'train1' and 'val1' (sequence length 2).

train_motion_sampling:
Condition: Always active.
Purpose: Trains the motion noise generator (self.mo_noise_generator) to produce appropriate noise for the motion update. Uses a kernel-based negative log-likelihood loss comparing the distribution of motion-updated particles to the true next state.
Iterators: Uses 'train1' and 'val1' (sequence length 2).

train_measurement_model:
Condition: Always active.
Purpose: Trains the observation encoder (self.encoder_conv, self.encoder_linear) and the observation likelihood estimator (self.obs_like_estimator). Uses a contrastive loss: it maximizes the likelihood for correct observation-state pairs (diagonal elements) and minimizes it for incorrect pairs (off-diagonal elements).
Iterators: Uses 'train1' and 'val1' (sequence length 2, although only the first step's observation/state are used in the loss).

train_particle_proposer:
Condition: Only active if self.use_proposer is True during initialization.
Purpose: Trains the particle proposer network (self.particle_proposer) to generate particles close to the true state given an observation. Uses a kernel-based negative log-likelihood loss comparing the distribution of proposed particles to the true state.
Iterators: Uses 'train1' and 'val1' (sequence length 2, only first step used).

train_e2e:
Condition: Always active.
Purpose: Trains the entire model end-to-end. It runs the full particle filter process (connect_modules) and calculates a kernel-based negative log-likelihood loss comparing the weighted particle cloud at each time step to the true state at that time step. It also monitors the MSE loss at the final time step.
Iterators: Uses 'train' and 'val' (full sequence length).
The fit method then executes these stages sequentially based on the train_individually and train_e2e flags passed to it, forming a training curriculum.