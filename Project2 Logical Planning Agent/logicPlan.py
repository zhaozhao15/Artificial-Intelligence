# logicPlan.py
# ------------
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


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""
import util
import sys
import logic
import game
import copy


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    ans1 = logic.Expr('A') | logic.Expr('B')
    ans2 = ~logic.Expr('A') % logic.disjoin([~logic.Expr('B'), logic.Expr('C')])
    ans3 = logic.disjoin([~logic.Expr('A'), ~logic.Expr('B'), logic.Expr('C')])
    return logic.conjoin([ans1, ans2, ans3])


def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    ans1 = logic.Expr('C') % logic.disjoin([logic.Expr('B'), logic.Expr('D')])
    ans2 = logic.Expr('A') >> logic.conjoin([~logic.Expr('B'), ~logic.Expr('D')])
    ans3 = ~logic.conjoin([logic.Expr('B'), ~logic.Expr('C')]) >> logic.Expr('A')
    ans4 = ~logic.Expr('D') >> logic.Expr('C')
    return logic.conjoin([ans1, ans2, ans3, ans4])


def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    WA0 = logic.PropSymbolExpr('WumpusAlive', 0)
    WA1 = logic.PropSymbolExpr('WumpusAlive', 1)
    WB0 = logic.PropSymbolExpr('WumpusBorn', 0)
    WK0 = logic.PropSymbolExpr('WumpusKilled', 0)
    ans1 = WA1 % logic.disjoin(logic.conjoin([WA0, ~WK0]), logic.conjoin([~WA0, WB0]))
    ans2 = ~logic.conjoin([WA0, WB0])
    return logic.conjoin([ans1, ans2, WB0])


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    CNF = logic.to_cnf(sentence)
    return logic.pycoSAT(CNF)


def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    return logic.disjoin(literals)


def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    list = []
    for i in range(0, len(literals)):
        for j in range(0, len(literals)):
            if i != j:
                list.append(logic.disjoin(~literals[i], ~literals[j]))
    return logic.conjoin(list)


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.conjoin(atLeastOne(literals), atMostOne(literals))


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    plan = []
    my_actions = []
    for i in model:
        my_string = str(i)
        tmp1 = my_string.index('[')
        tmp2 = my_string.index(']')
        if my_string[:tmp1] in actions:
            expr = logic.PropSymbolExpr(my_string[:tmp1], int(my_string[tmp1+1:tmp2]))
            if model[expr] == True:
                my_actions.append((my_string[:tmp1],int(my_string[tmp1+1:tmp2])))
    for i in range(len(my_actions)):
        for j in range (len(my_actions)):
            if my_actions[j][1] == i:
                plan.append(my_actions[j][0])
    return plan


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    print walls_grid
    list = []
    for i in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
        if not walls_grid[i[0]][i[1]]:
            last_pos = logic.PropSymbolExpr(pacman_str, i[0], i[1], t-1)
            if i[0] + 1 == x:
                action = logic.PropSymbolExpr('East', t-1)
            elif i[0] - 1 == x:
                action = logic.PropSymbolExpr('West', t - 1)
            elif i[1] + 1 == y:
                action = logic.PropSymbolExpr('North', t - 1)
            else:
                action = logic.PropSymbolExpr('South', t - 1)
            list.append(logic.conjoin(last_pos, action))
    return logic.PropSymbolExpr(pacman_str, x , y , t) % logic.disjoin(list)


def next_tick_goal(expr, goal, tick, walls):
    expr1 = copy.deepcopy(expr)
    expr1.append(pacmanSuccessorStateAxioms(goal[0], goal[1], tick + 1, walls))
    expr1.append(logic.PropSymbolExpr(pacman_str, goal[0], goal[1], tick + 1))
    model = findModel(logic.conjoin(expr1))
    if model:
        return extractActionSequence(model, ['North', 'South', 'East', 'West'])
    else:
        return None


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    goal = problem.getGoalState()
    start = problem.getStartState()
    expr = []
    expr.append(logic.PropSymbolExpr(pacman_str, start[0], start[1], 0))
    for i in range(1, width+1):
        for j in range(1, height+1):
            if (i, j) != start and not walls[i][j]:
                expr.append(~logic.PropSymbolExpr(pacman_str, i, j, 0))
    for tick in range(0, 51):
        for i in range(1, width + 1):
            for j in range(1, height + 1):
                if tick != 0 and not walls[i][j]:
                    expr.append(pacmanSuccessorStateAxioms(i, j, tick, walls))
        if tick != 0:
            tmp = exactlyOne([logic.PropSymbolExpr('North', tick-1), logic.PropSymbolExpr('South', tick-1), logic.PropSymbolExpr('East', tick-1), logic.PropSymbolExpr('West', tick-1)])
            expr.append(tmp)
            if next_tick_goal(expr, goal, tick, walls) != None:
                return next_tick_goal(expr, goal, tick, walls)
    return None


def next_tick_eat_all(expr, food, tick, width, height):
    expr1 = copy.deepcopy(expr)
    for i in range(1, width + 1):
        for j in range(1, height + 1):
            tmp_list = []
            if food[i][j]:
                for k in range(1, tick + 1):
                    tmp_list.append(logic.PropSymbolExpr(pacman_str, i, j, k))
                expr1.append(atLeastOne(tmp_list))
    model = findModel(logic.conjoin(expr1))
    if model:
        return extractActionSequence(model, ['North', 'South', 'East', 'West'])
    else:
        return None


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    start = problem.getStartState()[0]
    food = problem.getStartState()[1]
    expr = []
    expr.append(logic.PropSymbolExpr(pacman_str, start[0], start[1], 0))
    for i in range(1, width+1):
        for j in range(1, height+1):
            if (i, j) != start and not walls[i][j]:
                expr.append(~logic.PropSymbolExpr(pacman_str, i, j, 0))
    for tick in range(0, 51):
        for i in range(1, width + 1):
            for j in range(1, height + 1):
                if tick != 0 and not walls[i][j]:
                    expr.append(pacmanSuccessorStateAxioms(i, j, tick, walls))
        if tick != 0:
            tmp = exactlyOne([logic.PropSymbolExpr('North', tick - 1), logic.PropSymbolExpr('South', tick - 1), logic.PropSymbolExpr('East', tick - 1), logic.PropSymbolExpr('West', tick - 1)])
            expr.append(tmp)
            if next_tick_eat_all(expr, food, tick, width, height) != None:
                return next_tick_eat_all(expr, food, tick, width, height)
    return None


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    