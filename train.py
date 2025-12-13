import os
import torch
import math
import numpy as np
import random
import pickle
import gc
import multiprocessing as mp

from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from network import AZDualRes
from self_play import self_play_batch
from collections import deque


def perform_model_training(model: AZDualRes, training_set,
                                game_buffer: int, n_batches: int, batch_size: int, 
                                optimizer_parameters: dict, device: torch.device):
    """
    Update the already existing neural network using the training data which was generated from self-play.

    Loss Function: "The neural network is adjusted to minimize the error between the predicted value and the self-play
    winner, and to maximize the similarity of the neural network move probabilities to the search probabilities": The
    parameters are adjusted by gradient descent on a loss function that sums over the mean-squared error
    and cross-entropy losses. The cross-entropy and MSE losses are weighted equally. L2 weight regularization is
    used to prevent overfitting.

    Optimization: The neural network parameters are optimized by stochastic gradient descent with momentum (without
    learning rate annealing as opposed to the original paper).

    Parameters
    ----------
    model : AZNeuralNetwork
        the neural network which is updated
    training_set : [[[np.ndarray, np.ndarray, [float], float]]]
        list (per game) of list of training examples (s, p, v) (from the current player's POV)
    game_buffer : int
        max number of games to learn from
    n_batches : int
        number of batches to train off
    batch_size:
        number of examples per batch
    optimizer_parameters : dict
        optimiser hyperparameters
    device : torch.device
        device on which model training is performed
    """

    # used to select batches without duplicating entries
    all_indices = [(i, j) for i, game in enumerate(training_set) for j, _ in enumerate(game)]
    
    n_actions = len(training_set[0][0][1])
    game_size = int(-0.5 + math.sqrt(4 + 8 * n_actions) / 4)
    feature_size = 2 * game_size + 1

    print(f"Batches are sampled from {len(all_indices):,} training examples (incl. augmentations) from the "
            f"{len(training_set):,}/{game_buffer:,} most recent games.")

    # loss Functions and optimizer
    MSELoss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_parameters["learning_rate"],
        weight_decay=optimizer_parameters["weight_decay"]
    )

    print("Updating model .. ")
    # model update
    model.train()
    model.to(device)

    for i in tqdm(range(n_batches)):
        # process each batch 1 at a time to avoid memory overload
        batch_inds = random.sample(all_indices, batch_size)
        
        s_list = np.empty((batch_size, 8, feature_size, feature_size), dtype=np.float32)
        p_list = np.empty((batch_size, n_actions), dtype=np.float32)
        v_list = np.zeros((batch_size,), dtype=np.int8)

        for ind, (i, j) in enumerate(batch_inds):
            board, p, v = training_set[i][j]

            s = model.encode(board[0], board[1])
            s_list[ind] = s
            p_list[ind] = p
            v_list[ind] = v

        x = torch.tensor(s_list, dtype=torch.float32, device=device)
        p_gt = torch.tensor(p_list, dtype=torch.float32, device=device)
        v_gt = torch.tensor(v_list, dtype=torch.float32, device=device)
        
        optimizer.zero_grad()

        p, v = model.forward(x)

        # loss and model & optimizer update
        log_p = torch.log_softmax(p, dim=1)
        p_loss = -(p_gt * log_p).sum(dim=1).mean()
        v_loss = MSELoss(v, v_gt)
        loss = p_loss + v_loss
        loss.backward()
        optimizer.step()
        
        # free memory
        del x, p_gt, v_gt, p, v, loss, s_list, p_list, v_list
        torch.cuda.empty_cache()

    # evaluate model on same data
    print("Evaluating model .. ")
    model.eval()
    with torch.no_grad():

        # calculate loss per training example
        p_loss, v_loss = 0, 0
        for i in tqdm(range(n_batches)):
            optimizer.zero_grad()

            batch_inds = random.sample(all_indices, batch_size)
        
            s_list = np.empty((batch_size, 8, feature_size, feature_size), dtype=np.float32)
            p_list = np.empty((batch_size, n_actions), dtype=np.float32)
            v_list = np.zeros((batch_size,), dtype=np.int8)

            for ind, (i, j) in enumerate(batch_inds):
                board, p, v = training_set[i][j]

                s = model.encode(board[0], board[1])
                s_list[ind] = s
                p_list[ind] = p
                v_list[ind] = v

            x = torch.tensor(s_list, dtype=torch.float32, device=device)
            p_gt = torch.tensor(p_list, dtype=torch.float32, device=device)
            v_gt = torch.tensor(v_list, dtype=torch.float32, device=device)

            p, v = model.forward(x)
            log_p = torch.log_softmax(p, dim=1)
            p_loss += -(p_gt * log_p).sum(dim=1).mean()
            v_loss += MSELoss(v, v_gt)
            
            del x, p_gt, v_gt, p, v, s_list, p_list, v_list
            torch.cuda.empty_cache()

        p_loss = p_loss / n_batches
        v_loss = v_loss / n_batches

    print("Policy Loss: {0:.5f} (avg.)".format(p_loss))
    print("Value Loss: {0:.5f} (avg.)".format(v_loss))
    print("Loss: {0:.5f} (avg.)".format(p_loss + v_loss))

    return p_loss.item(), v_loss.item()

def train_model(config: dict, checkpoint_index):
    mp.set_start_method("spawn", force=True)
    
    training_set = []
    
    # load the model config and training set if they already exist
    folder_path = f'./models/{config["name"]}'
    if os.path.isdir(folder_path):
        with open(f"{folder_path}/config.pkl", 'rb') as f:
            config = pickle.load(f)
        with open(f"{folder_path}/training_set.pkl", 'rb') as f:
            training_set = pickle.load(f)
    else:
        os.mkdir(folder_path)
        with open(f'{folder_path}/config.pkl', 'wb') as f:
            pickle.dump(config, f)
            
        with open(f'{folder_path}/training_set.pkl', 'wb') as f:
            pickle.dump(training_set, f)
    
    training_set = deque(training_set, maxlen=config["game_buffer"])

    game_size = config["game_size"]
    n_iterations = config["n_iterations"]
    n_games = config["n_games"]
    game_buffer = config["game_buffer"]
    n_batches = config["n_batches"]
    batch_size = config["batch_size"]
    self_play_batch_size = config["self_play_batch_size"]

    mcts_parameters = config["mcts_parameters"]
    model_parameters = config["model_parameters"]
    optimizer_parameters = config["optimizer_parameters"]

    # use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the neural network
    nnet = AZDualRes(
        game_size=game_size,
        device=device,
        model_parameters=model_parameters,
    ).float()

    num_workers = 4
    
    self_play_batches = n_games // self_play_batch_size

    p_losses = []
    v_losses = []
    start_it = 1

    # load model state if it exists
    if checkpoint_index:
        checkpoint = torch.load(f"{folder_path}/checkpoint_{checkpoint_index}.pth", weights_only=False)
        
        nnet.load_state_dict(checkpoint['net_state_dict'])
        p_losses = checkpoint['p_losses']
        v_losses = checkpoint['v_losses']
        start_it = len(p_losses)+1

    # create a pool of subprocesses for multiprocessing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for iteration in range(start_it, n_iterations+1):
            print(f"\n#################### Iteration {iteration}/{n_iterations} #################### ")

            nnet.eval()
            nnet.to(device)

            print("------------ Self-Play using MCTS ------------")

            # define self-play arguments for subprocesses
            model_state = nnet.state_dict()
            worker_args = [(game_size, self_play_batch_size, model_state, device, model_parameters, mcts_parameters) for i in range(self_play_batches)]

            # submit the subprocess jobs
            futures = [executor.submit(self_play_batch, args) for args in worker_args]

            # as self-play batches are completed, add them to the training set
            for future in tqdm(as_completed(futures), total=len(futures)):
                batch_result = future.result()
                training_set.extend(batch_result)

                del batch_result
                gc.collect()
                torch.cuda.empty_cache()

            print("\n---------- Neural Network Training -----------")
            p_loss, v_loss = perform_model_training(
                model=nnet,
                training_set=training_set,
                game_buffer = game_buffer,
                n_batches = n_batches,
                batch_size = batch_size,
                optimizer_parameters=optimizer_parameters,
                device=device
            )
            nnet.to(device)
            torch.cuda.empty_cache()

            p_losses.append(p_loss)
            v_losses.append(v_loss)

            # update the checkpoint and save iteration results
            checkpoint = {
                'net_state_dict': nnet.state_dict(),
                'p_losses': p_losses,
                'v_losses': v_losses
            }

            gc.collect()

            torch.save(checkpoint, f'{folder_path}/checkpoint_{iteration}.pth')
            with open(f'{folder_path}/training_set.pkl', 'wb') as f:
                pickle.dump(list(training_set), f)