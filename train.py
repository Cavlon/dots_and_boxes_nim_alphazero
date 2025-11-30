import os
import torch
import numpy as np
import random
import pickle
import multiprocessing as mp
from tqdm.auto import tqdm
from network import AZDualRes
from tqdm.contrib.concurrent import process_map
from self_play import self_play_batch


def perform_model_training(model: AZDualRes, train_examples_per_game_augmented: list,
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
    learning rate annealingas opposed to the original paper).

    Parameters
    ----------
    model : AZNeuralNetwork
        the neural network which is updated
    train_examples_per_game_augmented : [[[np.ndarray, np.ndarray, [float], float]]]
        list (per game) of list of training examples (l, b, p, v) (from the current player's POV)
    data_parameters, optimizer_parameters : dict
        training hyperparameters
    device : torch.device
        device on which model training is performed
    """

    # sample specific number of batches
    print("Encoding train examples for given model .. ")
    train_examples = [t for t_list in train_examples_per_game_augmented for t in t_list]  # flatten training list
    train_examples = [(model.encode(board[0], board[1]), p, v) for board, p, v in train_examples]  # encode s=(l, b) for given model
    for s, p, v in train_examples:
        # for feature planes representation, batching s of shape [4, n, n] should result in batch of shape [batch_size, 4, n, n]
        # if no dimension is added, resulting shape would be [4*batch_size, n, n] which would ignore one necessary dimension
        s.shape = (1,) + s.shape
    print(f"Batches are sampled from {len(train_examples):,} training examples (incl. augmentations) from the "
            f"{len(train_examples_per_game_augmented):,}/{game_buffer:,} most recent games.")


    print("Preparing batches .. ")

    batches = []
    for _ in tqdm(range(n_batches)):
        batch = random.sample(train_examples, batch_size)
        x, p, v = [list(t) for t in zip(*batch)]
        batches.append((np.vstack(x), np.vstack(p), v))

    # loss Functions and optimizer
    CrossEntropyLoss = torch.nn.CrossEntropyLoss()
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
        optimizer.zero_grad()

        # to not run out of memory on gpu, move data to device sequentially
        x, p_gt, v_gt = [torch.tensor(e, dtype=torch.float32, device=device) for e in batches[i]]  # batch = (x, p, v)

        p, v = model.forward(x)

        # loss and model & optimizer update
        p_loss = CrossEntropyLoss(p, p_gt)
        v_loss = MSELoss(v, v_gt)
        loss = p_loss + v_loss
        loss.backward()
        optimizer.step()

    # evaluate model on same data
    print("Evaluating model .. ")
    model.eval()
    with torch.no_grad():

        # calculate loss per training example
        p_loss, v_loss = 0, 0
        for i in tqdm(range(n_batches)):
            optimizer.zero_grad()

            x, p_gt, v_gt = [torch.tensor(e, dtype=torch.float32, device=device) for e in batches[i]]  # batch = (x, p, v)

            p, v = model.forward(x)
            p_loss += CrossEntropyLoss(p, p_gt)
            v_loss += MSELoss(v, v_gt)

        p_loss = p_loss / n_batches
        v_loss = v_loss / n_batches

    print("Policy Loss: {0:.5f} (avg.)".format(p_loss))
    print("Value Loss: {0:.5f} (avg.)".format(v_loss))
    print("Loss: {0:.5f} (avg.)".format(p_loss + v_loss))

    return p_loss.item(), v_loss.item()

def train_model(config: dict, checkpoint_index):
    mp.set_start_method("spawn", force=True)
    
    training_set = []  # A list of lists where each list is a game's training data
    
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

    # Use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the neural network
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

    if checkpoint_index:
        checkpoint = torch.load(f"{folder_path}/checkpoint_{checkpoint_index}.pth", weights_only=False)
        
        nnet.load_state_dict(checkpoint['net_state_dict'])
        p_losses = checkpoint['p_losses']
        v_losses = checkpoint['v_losses']
        start_it = len(p_losses)+1

    for iteration in range(start_it, n_iterations+1):
        print(f"\n#################### Iteration {iteration}/{n_iterations} #################### ")

        nnet.eval()
        nnet.to(device)

        print("------------ Self-Play using MCTS ------------")

        worker_args = [(game_size, self_play_batch_size, nnet.state_dict(), device, model_parameters, mcts_parameters) for i in range(self_play_batches)]

        new_training_data = process_map(
            self_play_batch,
            worker_args,
            max_workers=num_workers,
            tqdm_class=tqdm,
            chunksize=1
        )

        for batch in new_training_data:
            training_set.extend(batch)

        while len(training_set) > game_buffer:
            training_set.pop(0)

        print("\n---------- Neural Network Training -----------")
        p_loss, v_loss = perform_model_training(
            model=nnet,
            train_examples_per_game_augmented=training_set,
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
        
        checkpoint = {
            'net_state_dict': nnet.state_dict(),
            'p_losses': p_losses,
            'v_losses': v_losses
        }

        torch.save(checkpoint, f'{folder_path}/checkpoint_{iteration}.pth')
        with open(f'{folder_path}/training_set.pkl', 'wb') as f:
            pickle.dump(training_set, f)