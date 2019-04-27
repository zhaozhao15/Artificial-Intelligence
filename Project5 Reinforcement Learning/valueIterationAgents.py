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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        cnt = 0
        while cnt < self.iterations:
            new = util.Counter()
            for state in self.mdp.getStates():
                q_values = util.Counter()

                if len(self.mdp.getPossibleActions(state)) == 0:
                    new[state] = 0
                    continue

                for action in self.mdp.getPossibleActions(state):
                    q_values[action] = self.computeQValueFromValues(state, action)
                my_max = -9999
                for i in q_values:
                    if q_values[i] > my_max:
                        my_max = q_values[i]
                new[state] = my_max

            cnt = cnt + 1
            self.values = new





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
        value = 0
        for next_state, p in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, next_state)
            value = value + p*(r + self.discount*self.values[next_state])
        return value



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.mdp.getPossibleActions(state)
        if len(legal_actions) == 0:
            return None
        max = -9999
        best_action = None
        for action in legal_actions:
            value = self.computeQValueFromValues(state, action)
            if value > max:
                max = value
                best_action = action
        return best_action



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
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
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
        cnt = 0
        while cnt < self.iterations:
            for state in self.mdp.getStates():
                if cnt < self.iterations:
                    if self.mdp.isTerminal(state):
                        self.values[state] = 0
                    elif len(self.mdp.getPossibleActions(state)) == 0:
                        self.values[state] = 0
                    else:
                        max = -9999
                        best_action = None
                        for action in self.mdp.getPossibleActions(state):
                            value = self.computeQValueFromValues(state, action)
                            if value > max:
                                max = value
                                best_action = action
                        self.values[state] = max
                cnt = cnt + 1



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > self.theta:
                        if nextState not in predecessors:
                            predecessors[nextState] = [state]
                        else:
                            predecessors[nextState].append(state)
        queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max = -9999
                for action in self.mdp.getPossibleActions(state):
                    value = self.computeQValueFromValues(state, action)
                    if value > max:
                        max = value
                diff = abs(self.values[state] - max)
                queue.push(state, -diff)

        cnt = 0
        while cnt < self.iterations:
            if queue.isEmpty():
                break

            state = queue.pop()
            if not self.mdp.isTerminal(state):
                max = -999999
                for action in self.mdp.getPossibleActions(state):
                    value = self.computeQValueFromValues(state, action)
                    if value > max:
                        max = value
                self.values[state] = max

            for p in predecessors[state]:
                max = -999999
                for action in self.mdp.getPossibleActions(p):
                    value = self.computeQValueFromValues(p, action)
                    if value > max:
                        max = value
                diff = abs(self.values[p] - max)
                if diff > self.theta:
                    queue.update(p, -diff)
            cnt += 1
