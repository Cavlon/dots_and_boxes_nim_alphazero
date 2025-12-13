import numpy as np
import copy
import torch

from network import AZDualRes
from game import DotsAndBoxesGame
from mcts import MCTS

def self_play(game_size: int, model: AZDualRes, mcts_parameters: dict, line_indices: np.ndarray):
    """
    Perform a single game of self-play using MCTS. The data for the game is stored as (s, p, v) at each
    time-step (i.e., each turn results in a training example), with s (board) representing the game state, 
    and p (policy vector) and v (value scalar) being the parameters to be predicted.

    Returns
    -------
    train_examples : [[(np.ndarray, np.ndarray), [float], float]]
        list of training examples (s, p, v) (from the current player's POV)
    """

    game = DotsAndBoxesGame(size=game_size)
    n_moves = 0
    train_examples = []

    # one self-play corresponds with one tree
    mcts = MCTS(
        model=model,
        game=copy.deepcopy(game),
        args=mcts_parameters
    )

    # temperature adds variation to starting moves
    temperature_move_threshold = mcts_parameters["temperature_move_threshold"]

    while game.result is None:
        temp = 1 if n_moves < temperature_move_threshold else 0
        n_moves += 1

        # execute MCTS for next move
        probs = mcts.getProbs(temp=temp)
        
        # add the board and all its symmetries to the training set
        canonical_board = game.getCanonicalBoard()
        symmetries = DotsAndBoxesGame.getSymmetries(canonical_board[0], canonical_board[1])
        policy_symmetries = DotsAndBoxesGame.getLineSymmetries(np.asarray(probs))
        
        train_examples.extend([[(l_sym, b_sym), p_sym, game.current_player] for l_sym, b_sym, p_sym in zip(symmetries[0], symmetries[1], policy_symmetries)])

        # sample and play move from probability distribution
        move = np.random.choice(line_indices, p=probs)
        game.playMove(move)

        # child node corresponding to the played action becomes the new root. The subtree below this child is
        # retained along with all its statistics, while the remainder of the tree is discarded
        mcts.root = mcts.root.children[move]

    # determine correct value v for the activate player in each example
    for i, (_, _, current_player) in enumerate(train_examples):       
        if current_player == game.result:
            train_examples[i][-1] = 1  # the player that made this move won
        elif game.result == 0:
            train_examples[i][-1] = 0  # the game drew
        else:
            train_examples[i][-1] = -1  # the player that made this move lost

    return train_examples

def self_play_batch(args):
    # initialise the neural network once and play a small batch of games with it (for multiprocessing)
    results = []
    
    game_size, batch_size, nnet_state, device, model_parameters, mcts_parameters = args
    line_indices = np.arange(DotsAndBoxesGame(game_size).board.N_LINES)
    
    nnet = AZDualRes(
        game_size=game_size,
        device=device,
        model_parameters=model_parameters,
    ).float()
    
    nnet.load_state_dict(nnet_state)
    nnet.eval()
    nnet.to(device)
    
    with torch.no_grad():
        for i in range(batch_size):
            results.append(self_play(game_size, nnet, mcts_parameters, line_indices))
    return results