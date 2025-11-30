import numpy as np
import copy
import math
import random
from typing import List
from game import DotsAndBoxesGame
from network import AZDualRes

class AZNode():
    """
    Implements the search tree of AlphaZero. Each node corresponds to a board s.

    Attributes
    ----------
    s : DotsAndBoxesGame
        the game state this node represents
    a : int
        move that was executed at the parent's position, resulting in this node's position s
    children : dict
        child nodes children[a] = child node after performing action a
    Q : dict
        action values Q[a] = Q(s,a)
    N : dict
        visit counts N[a] = N(s,a)
    P : np.ndarray
        action values P(s,a) as returned by the neural network
    """

    def __init__(self, parent, s: DotsAndBoxesGame, a: int):

        # if a node has a parent, it must have a move to get to this state
        assert (parent is None and a is None) or (parent is not None and isinstance(a, int))

        # create link with parent node
        if parent is not None:
            parent.children[a] = self

        self.a = a
        self.s = s

        self.children = {}
        self.Q = {}
        self.N = {}
        self.P = None

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: DotsAndBoxesGame, model: AZDualRes, args: dict):
        self.model = model
        self.root = AZNode(
            parent=None,
            a=None,
            s=game
        )
        
        self.num_simulations = args["n_simulations"]
        self.dirichlet_eps = args["dirichlet_eps"]
        self.dirichlet_alpha = args["dirichlet_alpha"]
        self.c_puct = args["c_puct"]

    def getProbs(self, temp: int = 1) -> List[float]:
        """
        (d) Play.
        Provides the core functionality of MCTS: output search probabilities recommending moves to play.

        Parameters
        ----------
        temp : int
            temperature controlling parameter

        Returns
        -------
        probs : [float]
            move probabilities pi(a) ~ N(s,a)^(1/temp)
        """
        
        s = self.root.s
        valid_moves = self.root.s.getValidMoves()
        
        # run simulations, add dirichlet noise for the root node's actions
        for i in range(self.num_simulations):
            dirichlet_noise = np.zeros((s.board.N_LINES,), dtype=np.float32)
            dirichlet_noise[valid_moves] = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))

            self.search(self.root, is_root=True, dirichlet_noise=dirichlet_noise)
        
        # make sure only valid moves are visited
        assert set(list(self.root.N.keys())).issubset(set(s.getValidMoves()))
        
        # determine how many times each valid action was taken
        counts = [self.root.N[a] if a in self.root.N else 0 for a in range(s.board.N_LINES)]
        
        # select the move with maximum visit count to give the strongest possible play (return value is one-hot vector)
        if temp == 0:
            probs = [0] * len(counts)
            probs[np.array(counts).argmax()] = 1
            return probs

        # pi(a) ~ N(s,a)^(1/temp) while ensuring a probability distribution
        probs = [n ** (1. / temp) for n in counts]
        probs = [p / float(sum(probs)) for p in probs]
        return probs

    def search(self, node: AZNode, is_root: bool = False, dirichlet_noise: np.ndarray = None) -> float:
        """
        Recursively perform a single simulation within MCTS.

        Parameters
        ----------
        node : AZNode
            node that corresponds to the MCTS's current position s
        is_root : bool
            whether the current node is the root node of the search or not (relevant for dirichlet noise)
        dirichlet_noise : np.ndarray
            dirichlet noise that is applied only on the prior probabilities of the root node

        Returns
        -------
        v : float
            probability of the current player winning in position s
        """

        if not node.s.result is None:
            # game is finished before reaching a non-visited node
            # return the actual score v for the current player
            # in case of a winner, current_player contains it (when capturing a box, the current player does not switch)
            return node.s.current_player * node.s.result

        # reached a leaf (node which was not visited yet)
        if node.P is None:
            return self.evaluate(node)

        # node was visited before, select an action to continue traversal
        a = self.select(node, is_root, dirichlet_noise)

        if a not in node.N:
            # applying the selected move means approaching a leaf (node which was not visited yet)
            child_s = copy.deepcopy(node.s)
            child_s.playMove(a)
            child = AZNode(
                parent=node,
                a=a,
                s=child_s
            )
        else:
            # get the node taking this action would entail
            child = node.children[a]

        # continue traversing, i.e., call method recursively
        v_child = self.search(child)

        # we now received a score v from the child node, either by ..
        # .. reaching a leaf (v in [-1,1] as calculated by the neural network) or by
        # .. finishing the game (v in {-1, 0, 1})
        # make sure to maintain the value depending on the player turn
        v = v_child if node.s.current_player == child.s.current_player else -v_child

        # backup before returning v
        if a not in node.N:
            # leaf: node was visited for the first time
            node.Q[a] = v
            node.N[a] = 1

        else:
            n = node.N[a]
            q = node.Q[a]
            node.Q[a] = (n * q + v) / (n + 1)  # Q = mean of v
            node.N[a] += 1

        return v
    
    def select(self, node: AZNode, is_root: bool, dirichlet_noise: np.ndarray) -> int:
        """
        (a) Select.
        Select the move with maximum action value Q, plus an upper confidence bound U that depends on a stored
        prior probability P and visit count N.

        Parameters
        ----------
        node : AZNode
            (non-leaf) node that corresponds to the MCTS's current position s
        is_root : bool
            whether the current node is the root node of the search or not (relevant for dirichlet noise)
        dirichlet_noise : bool
            dirichlet noise that is applied only on the prior probabilities of the root node

        Returns
        -------
        a_max : int
            move a for which Q(s,a) + U(s,a) is maximized
        """
        
        # make sure valid moves exist
        assert len(node.s.getValidMoves()) > 0

        maximum = float('-inf')
        a_max = -1

        N_sum = sum(node.N.values())
        N_sqrt = math.sqrt(N_sum)

        # add dirichlet noise to the root's probabilities
        P = node.P if not is_root else \
            (1 - self.dirichlet_eps) * node.P + self.dirichlet_eps * dirichlet_noise
        
        # make sure the probabilities sum to 1
        assert abs(np.sum(P) - 1) < 1e-6, \
            f"is_root: {is_root}, sum of P: {np.sum(node.P)}, sum of P after adding dirichlet noise: {np.sum(P)}"

        # find the action that maximises value
        for a in node.s.getValidMoves():
            # each move corresponds to a child node that may or may not have already been visited

            p = P[a]
            if a in node.N:
                q = node.Q[a]
                n = node.N[a]
            else:
                q = 0
                n = 0

            # upper confidence bound U(s, a) ~ P(s, a) / (1 + N(s, a))
            u = self.c_puct * p * N_sqrt / (1 + n)

            # maximize action value Q(s,a) + upper confidence bound U(s,a)
            if q + u > maximum:
                maximum = q + u
                a_max = a

        return a_max
    
    def evaluate(self, leaf: AZNode) -> float:
        """
        (b) (Expand and) Evaluate.
        Evaluate the associated position of the leaf node by the neural network
        and store the vector of P values.

        Parameters
        ----------
        leaf : AZNode
            (leaf) node that corresponds to the MCTS's current position s

        Returns
        -------
        v : float
            probability of the current player winning in position s
        """

        # represent the board from the perspective of the current player
        canonical_lines, canonical_boxes = leaf.s.getCanonicalBoard()

        # neural network evaluation is carried out on a reflection or rotation which is selected uniformly
        i = random.randint(0, 7)
        j = i
        if i == 1:
            j = 3
        elif i == 3:
            j = 1

        lines, boxes = DotsAndBoxesGame.getSymmetries(canonical_lines, canonical_boxes)
        lines = lines[i]
        boxes = boxes[i]

        # get prediction, and apply same revert rotation that was applied to lines vector before forwarding to neural network
        p, v = self.model.p_v(lines, boxes)
        p = DotsAndBoxesGame.getLineSymmetries(p)[j]

        leaf.P = p
        return v
    
    @staticmethod
    def determine_move(s: DotsAndBoxesGame, model: AZDualRes, mcts_parameters: dict) -> int:

        mcts = MCTS(s, model, mcts_parameters)
        probs = mcts.getProbs(
            temp=0  # select the move with maximum visit count, to give the strongest possible play
        )
        move = np.array(probs).argmax()

        valid_moves = np.where(s.board.l == 0)[0].tolist()
        assert move in valid_moves, f"move {move} is not a valid move in {valid_moves}"

        return move