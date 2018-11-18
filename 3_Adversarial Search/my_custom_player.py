
from sample_players import DataPlayer
import random
import math
from collections import defaultdict

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        #########################CHOOSE ONLY ONE##########################

        if False:
            #set my_heuristic to True to use advanced heuristic
            self.queue.put(self.alpha_beta(state, depth = 3, my_heuristic = False))
        if False: # Set True for iterative deepening with alpha_beta
            #set my_heuristic to True to use advanced heuristic
            self.id_alpha_beta(state, depth_limit = 20, my_heuristic = True)
        if True: # Set True for MCTS
            self.monte_carlo_tree_search(state, 2**(1/2))

    def monte_carlo_tree_search(self, state, c = 1):
        statistics = defaultdict(lambda: [0, 0]) #[X_j, n_j]
        # sub functions
        def tree_policy(state, statistics, c):
            nodes = [state]
            while state.terminal_test() == False:
                for action in state.actions():
                    if statistics[state.result(action)][1] == 0:
                        nodes.append(state.result(action))
                        return nodes
                nodes.append(state.result(select_action(state, statistics, c)))
                state = nodes[-1]
            return nodes

        def default_policy(state):
            while state.terminal_test() == False:
                state = state.result(random.choice(state.actions()))
            return state.utility_mcts(state.player()) * -1

        def select_action(state, statistics, c):
            return max(state.actions(), key = lambda x:
            statistics[state.result(x)][0]/(statistics[state.result(x)][1] + 0.01) +
            c * math.log(2*statistics[state][1])/(statistics[state.result(x)][1] + 0.01)**(1/2))

        def update(statistics, nodes, result):
            for node in reversed(nodes):
                statistics[node][0] += result
                statistics[node][1] += 1
                result = - result
            return statistics

        if state.ply_count < 2 and 57 in state.actions():
            # start with middle square if available and no opening book
            self.queue.put(57)
        else:
            # iteration
            while True:
                    nodes = tree_policy(state, statistics, c)
                    statistics = update(statistics, nodes, default_policy(nodes[-1]))
                    action = select_action(state, statistics, 0)
                    if action is not None:
                        self.queue.put(action)

    def id_alpha_beta(self, state, depth_limit = 20, my_heuristic = False):
        for d in range(1, depth_limit +1):
            action = self.alpha_beta(state, d, my_heuristic)
            if action is not None:
                self.queue.put(action)

    def alpha_beta(self, state, depth, my_heuristic):
        alpha = float("-inf")
        beta = float("inf")
        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state, my_heuristic)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha:
                        return value
                beta = min(beta,value)
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state, my_heuristic)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value
        if state.ply_count < 2 and 57 in state.actions():
            #start with center square if available
            return 57
        else:
            return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, alpha, beta))

    def score(self, state, my_heuristic):
        """
        Beginning of the game: Heuristic incentivizes proximity to opponent
        In the middle stage the game:Heuristic incentivizes proximity to middle
        in the end stage of the game: in the end stage of game heuristic incentivizes
            having more moves then opponent
        Heuristic only active when my_heuristic is set on True, otherwise it is
            the my_moves - opponent_moves heuristic
        """
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        my_moves = len(own_liberties)
        op_moves = len(opp_liberties)


        if my_heuristic:
            #Hard coding size of board here --> In slack it says this is fine
            coord_own = (own_loc % (11 + 2), own_loc // (11 + 2))
            coord_opp = (opp_loc % (11 + 2), opp_loc // (11 + 2))
            coord_mid = (5, 4)
            distance_mid = self.euclidean_distance(coord_own, coord_mid)
            distance_opp = self.euclidean_distance(coord_own, coord_opp)
            if state.ply_count < 30:
                #Beginning
                return my_moves - op_moves - distance_opp
            elif state.ply_count < 50:
                #Middle stage
                return my_moves - op_moves - distance_opp - distance_mid
            else:
                #End stage of game
                return my_moves - op_moves - distance_mid
        else:
        #mymoves - #opponentmoves heurestic for creating baseline performance
            return len(own_liberties) - len(opp_liberties)

    def euclidean_distance(self, coord1, coord2):
        """
        Calculates euclidean distance between to points.
        """
        return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**(1/2)

    def manhatten_distance(self, coord1, coord2):
        """
        Calculates manhatten distance between to points.
        """
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
