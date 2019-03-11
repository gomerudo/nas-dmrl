"""The DeepQ-Learning implementation for Atari games, from Deep Mind.

An adaptation I found on the internet.
"""

from collections import deque
import random
import warnings
import gym
import numpy as np
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.optimizers import RMSprop

warnings.filterwarnings('ignore')

# The first state is given by a reset operation (probably a predefined one)
# state = BD_ENV.reset()

# The shape of the environment is - apparently - a cube: 84x84x4
# Not sure what the 4 is, tho
ATARI_SHAPE = (84, 84, 4)
ACTION_SIZE = 3


def pre_process(frame_array):
    """Pre-process the image that is given as a frame-array (np.darray).

    The main operations are:
        a. Convert to gray scale (colors do not matter)
        b. resize the image to only 84x84 (the 3rd dimension does not matter)
        c. Return the resized image

    Args:
        frame_array (np.ndarray)    The image.

    Returns
        np.ndarray  The pre-processed image.

    """
    # converting into graysclae
    from skimage.color import rgb2gray
    grayscale_frame = rgb2gray(frame_array)

    # resizing the image
    from skimage.transform import resize
    resized_frame = np.uint8(
        resize(grayscale_frame, (84, 84), mode='constant') * 255
    )

    return resized_frame


def epsilon_greedy_policy_action(current_state, episode, epsilon):
    """Execute the epsilon-greedy strategy.

    This function will pick an action (as a number, i.e. its encoding), based
    on the e-greedy strategy, which can be sumarized as: randomly pick a
    number, if this number is lower than EPSILON, take a random action. Also,
    if the current episode is lower than the TOTAL_OBSERVE_COUNT (???),
    take a random action. Otherwise, take the best action based on the
    prediction of the Q-value for the current_state.

    Args:
        current_state (?)   The current state we are at.

    Returns:
        float   The ?

    """
    if np.random.rand() <= epsilon or episode < TOTAL_OBSERVE_COUNT:
        # take random action
        return random.randrange(ACTION_SIZE)
    else:
        # take the best actionm according to the predicted Q-value (I guess
        # from the Neural Net)
        Q_value = MODEL.predict(
            [current_state, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)]
        )
        return np.argmax(Q_value[0])


def huber_loss(y, q_value):
    """Compute the Huber's loss function.

    More details on this function at: https://en.wikipedia.org/wiki/Huber_loss.

    Args:
        y   (?) The predicted value.
        q_value (float) The predicted Q-value.

    Returns:
        float   The computed loss.

    """
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss


# deep mind model

def build_atari_model():
    """Build the Neural Network used for Q-learning.

    In Deep-Q-Learning we use a Neural Network (the <<deep>> part) to predict/
    compute the Q-values. The Neural Network (apparently the first implemented
    by Deep Mind) has the following structure (from input to output):

    - Input Layer for images
    - Input Layer for actions
    - Normalization Layer: Set image values to [0, 1] range, for RGB scale
    - Convolutional Layer: Uses ReLU
    - Convolutional Layer: Uses ReLU
    - Flatten convolution
    - Dense Layer of 256 units: Uses ReLU
    - Output Layer: Maps the 256 to n_actions
    - Filter Output Layer: Multiplies Output Layer with Input Layer for actions

    The model is defined as:
        input: Input Layer for images, Input Layer for actions
        output: Filter Output Layer.

    Returns:
        k.model The Neural Netork as a Keras model.

    """
    inputs = layers.Input(ATARI_SHAPE, name='inputs')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')

    normalized = layers.Lambda(lambda x: x / 255.0, name='norm')(inputs)

    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu')(normalized)
    conv_2 = layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    conv_flattened = layers.core.Flatten()(conv_2)
    hidden = layers.Dense(256, activation='relu')(conv_flattened)
    output = layers.Dense(ACTION_SIZE)(hidden)
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[inputs, actions_input], outputs=filtered_output)
    # model = \
    # Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)
    return model


# Constants for the Neural Network
MODEL = build_atari_model()  # Main model (temporal policy)
TARGET_MODEL = build_atari_model()  # Target model (real policy)
BATCH_SIZE = 32  # Batch size for Net's training


# Q-learning paremeters
N_EPISODES = 100000  # The number of episodes to consider
GAMMA = 0.99  # A parameter for Q-learning: discount factor
FINAL_EPSILON = 0.1  # The lower epsilon value we want to consider
EPSILON_STEP_NUM = 100000  # How many 'steps' we want for the epsilon strategy
EPSILON_DECAY = (1.0 - FINAL_EPSILON) / EPSILON_STEP_NUM  # Epsilon's decay
TOTAL_OBSERVE_COUNT = 750  # ????????
REPLAY_MEMORY = deque(maxlen=400000)  # The replay memory. Max 400K elements
TARGET_MODEL_CHANGE = 100  # How many episodes (?) to consider updating target


def get_sample_random_batch_from_replay_memory():
    """Obtain a random batch from the replay memory.

    The replay memory is the 'memory of good configurations', hence we return
    `BATCH_SIZE` (32) good configurations observed in the past.

    Returns:
        list    List of frames representing the current state.
        list    List of actions for each element in the batch.
        list    List of rewards for each element in the batch.
        list    List of termination flags for each element in the batch.
        list    List of frames representing the next state, per element.

    """
    # A minibatch is sampled, randomly, from the replay memory. This
    # instruction can be read as: Take bath_size (32) elements from the replay
    # memory, and assign them to mini_batch.
    mini_batch = random.sample(REPLAY_MEMORY, BATCH_SIZE)

    # Initialize empty ndarrays of shape 84x84x4 (the original input shape)
    current_state_batch = np.zeros((BATCH_SIZE, 84, 84, 4))
    next_state_batch = np.zeros((BATCH_SIZE, 84, 84, 4))

    # Create empty lists for the actions, rewards and termination flags.
    actions, rewards, dead = [], [], []

    # For each index and value of elements in the random sample (mini-batch)
    for idx, val in enumerate(mini_batch):
        # The memory replay has a list of 5-tuples (RL-variables):
        #   0: The current state
        #   1: The action taken
        #   2: The reward
        #   3: The next_state
        #   4: The termination flag
        current_state_batch[idx] = val[0]
        actions.append(val[1])
        rewards.append(val[2])
        next_state_batch[idx] = val[3]
        dead.append(val[4])

    return current_state_batch, actions, rewards, next_state_batch, dead


def deepQlearn():
    """Execute DeepQ-Learning algorithm (i.e. learn using DeepQ-Learning.

    This is basically an iteration of the Q-learning algorithm.

    The algorithm is as follows:
        1. Obtain a random batch from replay memory.
        2. Predict the Q-values of this batch, using target policy
        3. Compute the new Q-values of the minibatch, to fake the label
        4. Optimize the Neural Net, with respect to the fake labels.

    """
    # obtain a minibatch from the sample memory, with all needed information
    current_state_batch, actions, rewards, next_state_batch, dead = \
        get_sample_random_batch_from_replay_memory()

    # Create matrix, filled with 1s, of shape (BATCH_SIZE=32, ACTIONS_SIZE)
    # This is, each state will have a vector of actions.
    actions_mask = np.ones((BATCH_SIZE, ACTION_SIZE))

    # We use the target model to predict, not the 'temporal' model.
    next_Q_values = TARGET_MODEL.predict([next_state_batch, actions_mask])

    # We build an array of targets, of size BATCH_SIZE=32, this is, the
    # "actual" labels.
    targets = np.zeros((BATCH_SIZE,))

    # For every element (index, more precisely) in the BATCH_SIZE
    for i in range(BATCH_SIZE):
        # If the termination flag is 1, then the target of that state is -1
        # why -1? I have no idea...
        if dead[i]:
            targets[i] = -1
        # Otherwise, if termination flag is 0, compute Bellman's equation
        else:
            targets[i] = rewards[i] + GAMMA * np.amax(next_Q_values[i])

    # Convert actions and targets to one-hot vectors
    one_hot_actions = np.eye(ACTION_SIZE)[np.array(actions).reshape(-1)]
    one_hot_targets = one_hot_actions * targets[:, None]

    # Train the model with the mini-batch that came from replay memory.
    MODEL.fit(
        [current_state_batch, one_hot_actions],  # The state-action pairs
        one_hot_targets,  # We optimize with respect to targets
        epochs=1,  # Just make one iteration, i.e. one update
        batch_size=BATCH_SIZE,  # The batch size is clearly 32: all elements
        verbose=2  # no log information from training... maybe change it...
    )


def save_model(episode):
    """Save the atari model (the 'temporal') in the typical h5 format."""
    model_name = "atari_model{}.h5".format(episode)
    MODEL.save(model_name)


def get_one_hot(targets, nb_classes):
    """Obtain the One-Hot representation of the targets.

    Args:
        targets (np.array) The input to convert.
        nb_clases (int) The number of classes

    Returns:
        np.ndarray The converted array, as one-hot representations
                   (more dimensions).

    """
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def solve_atari():
    """Solve the atari problem."""
    # THIS IS THE ACTUAL ALGORITHM
    # We are using the Breakout Atari game, in its deterministic form
    bd_env = gym.make('BreakoutDeterministic-v4')
    epsilon = 1.0  # The initial value of epsilon
    max_score = 0  # ?????

    # Perform all episodes (every episode is a new game)
    for episode in range(N_EPISODES):
        # Propertires of the game at the start of every episode:
        #   a. We start the game alive
        #   b. We are not done when we start
        #   c. We have 5 lives given at the beginning
        #   d. We have score=0 at start
        dead, done, lives_remaining, score = False, False, 5, 0

        # The initial state at every episode is the default one, given by reset
        current_state = bd_env.reset()
        bd_env.render()

        # First, we perform a random number number of actions of type 1
        for _ in range(random.randint(1, 30)):
            current_state, _, _, _ = bd_env.step(1)

        # Get the image reduced to grey scale (2D only)
        current_state = pre_process(current_state)

        # Basically, make the current step a volume of repeated frames
        # But why? ... I don't know ...
        current_state = np.stack(
            (current_state, current_state, current_state, current_state),
            axis=2
        )
        current_state = np.reshape([current_state], (1, 84, 84, 4))

        # Perform Q-Learning procedure until the game indicates we are done:
        # either winning, or completely dying (no more lives)
        while not done:
            bd_env.render()
            # Take an action, according to the epsilon greedy strategy
            action = epsilon_greedy_policy_action(
                current_state,
                episode,
                epsilon)
            # Shift the selection, since action is from [0, N_ACTIONS) and we
            # want it from [1, N_ACTIONS]
            real_action = action + 1

            # If the epsilon is still bigger than what we defined at the
            # beginning and the episode is bigger that the TOTAL_OBSERVE_COUNT:
            # update the epislon with a value decrease
            if epsilon > FINAL_EPSILON and episode > TOTAL_OBSERVE_COUNT:
                epsilon -= EPSILON_DECAY

            # We perform the action and obtain the useful information
            next_state, reward, done, lives_left = bd_env.step(real_action)

            # We obtain the grey-scale version of the next_state and create the
            # volume as we did with current, but with only 1 element in the
            # 4th dimension
            next_state = pre_process(next_state)
            next_state = np.reshape([next_state], (1, 84, 84, 1))

            # We make the next state a volume of 4 frames, being the first one
            # the actual next_state and the remaining 3 the previous (current)
            # state
            next_state = np.append(
                next_state, current_state[:, :, :, :3],
                axis=3)

            # If after performing the action, we decrease the lives of the
            # environment, update the count and indicated we died, so that we
            # can resample an action from the last 'alive' state.
            if lives_remaining > lives_left['ale.lives']:
                dead = True
                lives_remaining = lives_left['ale.lives']

            # We add the current_state and its associated information, to the
            # replay memory
            REPLAY_MEMORY.append(
                (current_state, action, reward, next_state, dead)
            )

            # If we finally passed the first observe_count threshold, we start
            # learning
            if episode > TOTAL_OBSERVE_COUNT:
                # Perform 1 step of the Q-Learning policy update
                deepQlearn()

                # Update the target model if we reached the TARGET_MODEL_CHANGE
                # criterion.
                if episode % TARGET_MODEL_CHANGE == 0:
                    TARGET_MODEL.set_weights(MODEL.get_weights())

            # The score in the game is the accumulated reward
            score += reward

            # If we are dead after performing the action, we "revive" cause it
            # is only one live that we consumed.
            if dead:
                dead = False
            # otherwise, we keep playing
            else:
                current_state = next_state

            # If we overpass the max_score, update it
            if max_score < score:
                print(
                    "max score for the episode {} is : {} ".format(
                        episode,
                        score)
                )
                max_score = score

        # This is just a help procedure to back up every 100 episodes. It does
        # not have anything to do with the logic of the Q-learning procedure.
        if episode % 100 == 0:
            print(
                "final score for the episode {} is : {} ".format(
                    episode,
                    score)
                )
            save_model(episode)

# Solve the problem
# solve_atari()
