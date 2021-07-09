# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for itr in range(self.iterations):
            state_count = util.Counter()
            states = self.mdp.getStates()
            for state in states:
                max_val = float('-inf')
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    qval = self.computeQValueFromValues(state, action)
                    max_val = max(max_val, qval)
                    state_count[state] = max_val
            self.values = state_count

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        qval = 0
        for s_prime, prob in transitions:
            reward = self.mdp.getReward(state, action, s_prime)
            nxt_qval = self.values[s_prime]
            qval += prob * (reward + self.discount * nxt_qval)
        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            self.values[state] = 0
            return None
        v = float("-inf")
        policy = None
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            q_val = self.computeQValueFromValues(state, action)
            if q_val > v:
                v = q_val
                policy = action
        return policy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for i in range(self.iterations):
            indx = i % len(states)
            state = states[indx]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                actions_q_value = []
                for action in actions:
                    actions_q_value.append(self.getQValue(state, action))
                actions_q_value.sort()
                maximum_value = actions_q_value[-1]
                self.values[state] = maximum_value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        pq = util.PriorityQueue()
        predecessors = [[]] * len(states)

        for state in states:
            actions = self.mdp.getPossibleActions(state)

            for action in actions:
                next_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)

                for next_state, prob in next_states_and_probs:
                    if prob:
                        index = states.index(next_state)
                        if state not in predecessors[index]:
                            predecessors[index].append(state)

        for s in states:
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                q_values = [self.computeQValueFromValues(s, a) for a in actions]
                q_values.sort()
                max_q_value = q_values[-1]
                diff = abs(self.values[s] - max_q_value)
                pq.update(s, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                return
            s = pq.pop()
            index_of_s = states.index(s)

            actions = self.mdp.getPossibleActions(s)
            q_values = [self.computeQValueFromValues(s, a) for a in actions]
            q_values.sort()
            self.values[s] = q_values[-1]

            for p in predecessors[index_of_s]:
                moves = self.mdp.getPossibleActions(p)
                p_values = [self.computeQValueFromValues(p, a) for a in moves]
                p_values.sort()
                max_p_value = p_values[-1]
                diff = abs(max_p_value - self.values[p])
                if diff > self.theta:
                    pq.update(p, -diff)
