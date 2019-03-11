"""Dynamic Programming iterations for Reinforcement Learning: Q-Learning.

In this approach, we use a table of size `n_states*n_actions` that helps to
model the environment. More concretely, we will use two tables of the same
shape: one for the policy (which will contain the algorithm to move in the
environment) and one for the V-function. In practice, however, the V-function
can be programmed as a simple array of size `n_states`, although theoretically
it uses `n_states*n_actions` pieces of information.

In this example we implement Q-learning with two variants: a) policy iteration,
and b) value iteration.

**Value iteration**
It tries to model the V-function in one shot, i.e. obtain its optimal value or
its best approximation. Based on that, it will derive the optimal policy.

**Policy iteration**
It will iterate a number of EPOCHES to combine V-function improvements
(optimization of the V-function via policy evaluation) and policy improvements
(derivation of the policy using the latest discovered V-function).
"""

import numpy as np

from nas_rl2.rl2.rl.gridworld import GridworldEnv

GAMMA = 1.0  # The discount factor
GW_ENV = GridworldEnv()


def policy_evaluation(policy):
    """Perform the policy evaluation step.

    The goal is to create a V-function out of the policy.

    We first create an array of size `n_states`, which will serve as the
    V- function. Here, the V-function will store the `total_state_value`, which
    is the sum of Q-values for the actions associated to the state.

    REPEAT UNTIL CONVERGENCE:
        1. For each state `s`
            a. Obtain the row of probabilities in the policy `pi` for state `s`
            b. For every action-probability pair compute Bellman's equation and
            sum up the results into one single `total_state_value`.
            c. Update V[s]=total_state_value.
            d. Compute the delta.

    Convergence criterion here is the actual change of V[s] with respect to new
    total_state_value.

    Returns:
        np.array    The V-function

    """
    # In dynamic programming approaches we store a V matrix, that is the matrix
    # in charge of storing our estimations of the `value function`.
    # One per state.
    V = np.zeros(GW_ENV.nS)

    while True:
        # `delta` is a change we compute that will indicate whether or not
        # the cycle should terminate.
        delta = 0

        # In this two for loops, we are iterating over all elements in the
        # policy, i.e. for each state, we do something for each possible action

        # Here, we iterate each of the possible states (here, 16 states).
        # TODO: This is rather innefficient in terms of memory. We can use a
        #       simple counter in this for-loop.
        for state_index in range(GW_ENV.nS):
            # For each state, we compute a value, based on the actions
            total_state_value = 0

            # Now, for each possible action... (4 actions in our case)
            for action_index, prob_a in enumerate(policy[state_index]):
                # Obtain the Probability of the sate, the next state, the
                # reward, and ignore the termination flag.
                prob_s, next_state, reward, _ = \
                    GW_ENV.P[state_index][action_index]
                # We accumulate the total_state_value with the equation
                # P_a * P_s * (reward + Gamma *V[])
                total_state_value += \
                    prob_a * prob_s * (reward + GAMMA * V[next_state])

            # calculate change
            # the max makes sure that we update only if the new value is higher
            delta = max(delta, np.abs(total_state_value - V[state_index]))

            # We update the V's state cell with the latest total_state_value.
            V[state_index] = total_state_value

        # After evaluating all the states, we check if it is acceptable to make
        # one more iteration. Based on the `delta`.
        if delta < 0.005:
            break

    return np.array(V)


def policy_improvement(V, policy):
    """Improve the policy given a value function.

    The goal here is to fix V and update policy `pi`.

    REPEAT ONLY ONCE:
        1. For every state `s`:
            a. Compute the actions' the Q-values with Bellman's equation
            b. Obtain the best action: the one with the highest Q-value.
            c. Update the policy with a one-hot vector indicated the best
               action. # TODO: I don't know if this is correct... Does not make
               too much sense if we were working with action probs before.
    """
    # We again, iterate all the states, and per each of them, iterate all the
    # possible actions.
    for state_index in range(GW_ENV.nS):
        # We create the Q-value-array per each state, containing all actions
        Q_sa = np.zeros(GW_ENV.nA)

        # For each action, we will again obtain prob_s, next_state and reward.
        for action_index in range(GW_ENV.nA):
            # Obtain the Probability of the sate, the next state, the reward,
            # and ignore the termination flag.
            prob_s, next_state, reward, _ = \
                                        GW_ENV.P[state_index][action_index]
            # And compute the equation:
            # prob_state * (reward * GAMMA * V[next_state])
            Q_sa[action_index] += prob_s * (reward + GAMMA * V[next_state])

        # We obtain index of the maximum Q-value for the given state.
        best_action = np.argmax(Q_sa)
        # Update the policy. For the given state_index, it will store a dummy
        # row where 1 is set for the `best_action` index.
        policy[state_index] = np.eye(GW_ENV.nA)[best_action]

    return policy


def optimal_value_function(V):
    """Compute the optimal form of a given value-function V.

    The procedure is summarized as follow.

    REPEAT UNTIL CONVERGENCE:
        1. For every state `s`
            a. Obtain the Q-value of all possible actions (following Bellman's
            equation).
            b. Pick the highest Q-value `q` (i.e. the Q-value of the best
            action).
            c. Update V[s]=q (i.e. the latest estimated Q-value)

    Convergence criterion is the difference (delta) between the previous and
    the new V[s] (by default delta=0.00001).

    The idea behind is that, knowing the transition probability, we estimate
    the "goodness" of performing an action, based on the Q-value of the next
    state and the reward given by the environment. In this way, we learn with
    our estimated knowledge (of the "future") and the actual performance of the
    actions - until we approximate well our estimations, according to the delta
    criterion.

    Intuitively, the value function V represents the benefit of being at each
    state (somehow the higher the better, I guess...).

    Args:
        V (np.array) The value function to use in the process.

    Returns
        np.array The optimized V(alue)-function.

    """
    # We make the iteration, which will stop only if a given value delta is
    # reached.
    while True:
        # Initially, delta=0
        delta = 0

        # We will iterate over all states, and for each of them, we will
        # iterate over all the possible actions.
        for state_index in range(GW_ENV.nS):
            # The Q_sa is the Q-value array/list for all actions available.
            Q_sa = np.zeros(GW_ENV.nA)

            # For each action, we will compute the corresponding Q-value update
            for action_index in range(GW_ENV.nA):
                # Obtain the Probability of the sate, the next state, the
                # reward, and ignore the termination flag.
                prob_s, next_state, reward, _ = \
                                        GW_ENV.P[state_index][action_index]
                # We update the Q-value with the Bellman's equation:
                #   Prob_state * (reward + gamma*V[next_state])
                # A quick note that next_state is basically the realization of
                # where we will end up at, once we perform the action in
                # our current state.
                Q_sa[action_index] += \
                    prob_s * (reward + GAMMA * V[next_state])

            # Now, we will obtain the maximum Q-value for the current state,
            # i.e. the best candidate action.
            max_value_function_s = np.max(Q_sa)

            # TODO: not sure if this computation of delta is correct.
            # Delta is basically the max Q-value obtained minus the previous
            # value in the V-function, hence the actual change.
            delta = max(delta, np.abs(max_value_function_s - V[state_index]))

            # The v-function will now contain the maximum value.
            V[state_index] = max_value_function_s

        # Before continuing, check if the delta is big enough to continue.
        if delta < 0.00001:
            break

    # Return the optimized V-function.
    return V


def optimal_policy_extraction(V):
    """Exctract the best policy by following a given value function V.

    This is a similar algorithm to the optimal value function computation (see
    function `optimal_value_function(V)`). Now, however, the focus is in
    creating the policy - hence, maintaining a matrix of shape
    `(n_states*n_actions)`. This matrix will store, per each state, a one-hot
    vector (or dummy variable) with `1` at the position of the best possible
    action for that state.

    The V function is now fixed, meaning that we assume that we know everything
    it is needed - or at least a good approximation - about the benefit of each
    state.

    The algorithm is as follows.

    REPEAT ONLY ONCE:
        1. Create an empty policy `P` of size `n_states*n_actions`
        2. For each state `s`
            a. Compute again the Q-value of every action using Bellman's
            equation.
            b. Pick the action (index) with the highest Q-value
            c. Create the one-hot vector for that state and assign it to the
            corresponding position in the policy `P`.

    Therefore, the policy is telling us where to move (deterministically) at
    every position.

    Args:
        V (np.array) The value function to use in the process.

    Returns:
        np.darray   The policy inferred.

    """
    # Originally, the policy is a matrix of shape n_states*n_actions filled
    # with 0s.
    policy = np.zeros([GW_ENV.nS, GW_ENV.nA])

    # We will iterate over all states, and for each state we will iterate all
    # possible actions.
    for state_index in range(GW_ENV.nS):
        # The q-value is an array of size n_actions. We will use it to navigate
        # the environment by using the V-function.
        Q_sa = np.zeros(GW_ENV.nA)

        # We iterate every possible action in the environment.
        for action_index in range(GW_ENV.nA):
            # Obtain the Probability of the sate, the next state, the reward,
            # and ignore the termination flag.
            prob_s, next_state, reward, _ = \
                                        GW_ENV.P[state_index][action_index]
            # We update the Q-value of the action (not the state) with eq:
            # P_s * (reward + gamma*V[next])
            Q_sa[action_index] += prob_s * (reward + GAMMA * V[next_state])

        # We now pick the best q-value.
        best_action = np.argmax(Q_sa)
        # And we create a dummy array, where the index of the action with the
        # highest Q-value is marked with 1. The policy will store this.
        policy[state_index] = np.eye(GW_ENV.nA)[best_action]

    return policy


def policy_iteration():
    """Perform policy iteration algorithm.

    In this algorithm, we start by creating an empty policy `pi` of size
    `n_states*n_actions`, which will contain, per state, an array of
    uniformally distributed probabilities. This is, our policy is an array of
    probabilities that will rank, in a way, the actions.

    The improvement of such probabilities is achieved by performing a so-called
    evaluation/improvement cycle. In the evaluation phase...., wheres in the
    improvement phase....

    REPEAT FOR N_EPOCHS OR UNTIL CONVERGENCE:
        1. Evaluation: evaluates the policy `pi` to derive a V-function.
        2. Improvement: Using the V-function, improve the policy `pi`.

    Convergence is achieved when the improvement resulted in the same policy.

    Returns:
        np.darray The policy
        np.array  The last inferred V-function.

    """
    # Start with a random policy. A policy is a matrix of size
    # n_states*n_actions. Initially, the values are 1/n_actions, meaning that
    # every action at every state is equally probable.
    policy = np.ones([GW_ENV.nS, GW_ENV.nA]) / GW_ENV.nA

    # This is the number of episodes we want to train our agent.
    epochs = 1000
    # We iterate for all the numner of epoches. We will only stop earlier if we
    # find that convergence has been achieved.
    for i in range(epochs):
        # We evaluate the policy. The returned value in this step is the
        # state-action value table for the policy.
        V = policy_evaluation(policy)

        # We now "backup" the current policy.
        bkp_policy = np.copy(policy)

        # And we improve the policy, based on the state-action values
        improved_policy = policy_improvement(V, bkp_policy)

        # We check if the policy has changed. If it has not, it means
        # convergence has been achieved and we break the loop.
        if np.all(policy == improved_policy):
            print('Policy-Iteration converged at step {step}'.format(step=i+1))
            break

        # If no convergence, we subsitute the policy with the improved one.
        policy = improved_policy

    # Return the last policy considered, and the state-value function.
    return policy, V


def value_iteration():
    """Perform the value-iteration computation.

    In short, we need to create an empty V function (filled with 0's) and
    compute its optimal approximation. Later on, using that approximation, just
    infer the policy with a simple algorithm that picks the action with the
    highest Q-value per state.

    Returns:
        np.darray   The optimal policy.
        np.array    The optimal value function.

    """
    # In dynamic programming approaches we store a V matrix, that is the matrix
    # in charge of storing our estimations of the `value function`.
    # One per state.
    V = np.zeros(GW_ENV.nS)

    # The optimal value function is computed based on the initial V. The
    # functions returns the optimized version of V.
    optimal_v = optimal_value_function(V)

    # Once we have the optimial value function, we can compute the policy,
    # based on those V-values.
    policy = optimal_policy_extraction(optimal_v)

    # And finally return, as usual, the policy discovered and the V-function.
    return policy, optimal_v
