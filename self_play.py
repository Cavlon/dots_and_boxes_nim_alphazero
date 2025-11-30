import numpy as np
import copy
from network import AZDualRes
from game import DotsAndBoxesGame
from mcts import MCTS

def perform_self_play(game_size: int, model: AZDualRes, mcts_parameters: dict):
    """
    Perform a single game of self-play using MCTS. The data for the game is stored as (l, b, p, v) at each
    time-step (i.e., each turn results in a training example), with l (lines vector) and b (boxes matrix)
    representing the game state, and p (policy vector) and v (value scalar) being the parameters to be predicted.

    Returns
    -------
    train_examples : [[np.ndarray, np.ndarray, [float], float]]
        list of training examples (l, b, p, v) (from the current player's POV)
    """

    game = DotsAndBoxesGame(game_size)
    n_moves = 0
    train_examples = []

    # one self-play corresponds with one tree
    mcts = MCTS(
        model=model,
        game=copy.deepcopy(game),
        args=mcts_parameters
    )

    # when more than temperature_move_threshold moves were performed during self-play, the temperature parameter
    # is set from 1 to 0. This ensures that a diverse set of positions are encountered, as then the first moves
    # during MCTS are selected proportionally to their visit count
    temperature_move_threshold = mcts_parameters["temperature_move_threshold"]

    # iteration over time-steps t during the game. At each time-step, a MCTS is executed using the previous iteration
    # of the neural network and a move is played by sampling the search probabilities
    while game.result is None:
        temp = 1 if n_moves < temperature_move_threshold else 0
        n_moves += 1

        # execute MCTS for next move
        probs = mcts.getProbs(temp=temp)

        train_examples.append([
            game.getCanonicalBoard(),
            probs,
            game.current_player  # correct v is determined later
        ])

        # sample and play move from probability distribution
        move = np.random.choice(
            a=list(range(game.board.N_LINES)),
            p=probs
        )
        game.playMove(move)

        # child node corresponding to the played action becomes the new root. The subtree below this child is
        # retained along with all its statistics, while the remainder of the tree is discarded
        mcts.root = mcts.root.children[move]

    # determine correct value v for the activate player in each example
    for i, (_, _, current_player) in enumerate(train_examples):
        if current_player == game.result:
            train_examples[i][-1] = 1  # the player that made this move won
        else:
            train_examples[i][-1] = -1  # the player that made this move lost

    return train_examples

def self_play(game_size, nnet, mcts_parameters):

    new_game = perform_self_play(game_size, nnet, mcts_parameters)  # play a game and get training data
    new_game_augmented = []

    for board, p, v in new_game:  # get symmetries for each turn
        symmetries = DotsAndBoxesGame.getSymmetries(board[0], board[1])
        policy_symmeteries = DotsAndBoxesGame.getLineSymmetries(np.asarray(p))

        for i in range(8):
            new_game_augmented.append([(symmetries[0][i], symmetries[1][i]), policy_symmeteries[i], v])
    
    return new_game_augmented

def self_play_batch(args):
    results = []
    
    game_size, batch_size, nnet_state, device, model_parameters, mcts_parameters = args
    
    nnet = AZDualRes(
        game_size=game_size,
        device=device,
        model_parameters=model_parameters,
    ).float()
    
    nnet.load_state_dict(nnet_state)
    nnet.eval()
    nnet.to(device)
    
    for i in range(batch_size):
        results.append(self_play(game_size, nnet, mcts_parameters))
    return results