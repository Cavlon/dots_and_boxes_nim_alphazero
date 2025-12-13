import numpy as np
import torch
from torch import nn
from typing import Tuple
from game import Board

class AZDualRes(nn.Module):

    def __init__(self, game_size: int, device: torch.device, model_parameters: dict):
        super(AZDualRes, self).__init__()
        
        self.device = device
        self.game_size = game_size

        img_size = 2 * game_size + 1

        # use parameter information from config
        blocks_params = model_parameters["blocks"]
        residual_blocks = blocks_params["residual_blocks"] # number of residual blocks
        channels = blocks_params["channels"] # number of channels for residual blocks
        conv_kernel_size = blocks_params["conv_kernel_size"] # convolution kernel size
        res_kernel_size = blocks_params["res_kernel_size"] # residual kernel size
        stride = blocks_params["stride"] # convolution stride
        padding = blocks_params["padding"] # convolution padding

        heads_params = model_parameters["heads"]
        policy_head_channels = heads_params["policy_head_channels"] # policy convolution output channels
        value_head_channels = heads_params["value_head_channels"] # value convolution output channels
        heads_kernel_size = heads_params["heads_kernel_size"] # policy convolution output channels
        heads_stride = heads_params["heads_stride"] # convolution stride
        heads_padding = heads_params["heads_padding"] # convolution padding

        # convolutional block
        self.conv_block = ConvBlock(
            out_channels=channels,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=padding
        )

        # residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResBlock(
                n_channels=channels,
                kernel_size=res_kernel_size,
                stride=stride,
                padding=padding
            ) for _ in range(residual_blocks)]
        )

        # policy head
        self.policy_head = PolicyHead(
            conv_in_channels=channels,
            conv_out_channels=policy_head_channels,
            kernel_size=heads_kernel_size,
            stride=heads_stride,
            padding=heads_padding,
            fc_in_features=(policy_head_channels * img_size * img_size), # extract all values from all channels
            fc_out_features=(2 * self.game_size * (self.game_size + 1))  # dimension of policy vector
        )

        # value head (dimension=1 for resulting value)
        self.value_head = ValueHead(
            conv_in_channels=channels,
            conv_out_channels=value_head_channels,
            kernel_size=heads_kernel_size,
            stride=heads_stride,
            padding=heads_padding,
            fc_in_features=(value_head_channels * img_size * img_size) # extract all values from all channels
        )

        # initialize weights
        self.weight_init()
        self.float()


    def weight_init(self):
        """initialize model weights, xavier initialisation for weights, 0.01 for biases"""

        # conv block
        conv2d = self.conv_block.conv
        nn.init.xavier_normal_(conv2d.weight)
        conv2d.bias.data.fill_(0.01)

        # residual blocks
        for res_block in self.residual_blocks:
            # conv1
            nn.init.xavier_normal_(res_block.conv1.weight)
            res_block.conv1.bias.data.fill_(0.01)
            # conv2
            nn.init.xavier_normal_(res_block.conv2.weight)
            res_block.conv2.bias.data.fill_(0.01)

        # policy head
        nn.init.xavier_normal_(self.policy_head.conv.weight)
        self.policy_head.conv.bias.data.fill_(0.01)
        # fc
        nn.init.xavier_normal_(self.policy_head.fc.weight)
        self.policy_head.fc.bias.data.fill_(0.01)

        # value head
        nn.init.xavier_normal_(self.value_head.conv.weight)
        self.value_head.conv.bias.data.fill_(0.01)
        # fc1
        nn.init.xavier_normal_(self.value_head.fc1.weight)
        self.value_head.fc1.bias.data.fill_(0.01)
        # fc2
        nn.init.xavier_normal_(self.value_head.fc2.weight)
        self.value_head.fc2.bias.data.fill_(0.01)

    @staticmethod
    def encode(l: np.ndarray, b: np.ndarray) -> np.ndarray:
        """generate feature planes from lines and boxes (from the current player's perspective)"""

        game_size = Board.n_lines_to_size(l.size)
        img_size = 2 * game_size + 1

        # 1) full 0 layer for contrast
        img_0 = np.zeros((img_size, img_size), dtype=np.float32)
        # 2) full 1 layer for contrast
        img_1 = np.ones((img_size, img_size), dtype=np.float32)
        
        img_edges = np.zeros((img_size, img_size), dtype=np.float32)
        img_boxes = np.zeros((img_size, img_size), dtype=np.float32)
        
        img_taken_edges = np.zeros((img_size, img_size), dtype=np.float32)
        img_b_player = np.zeros((img_size, img_size), dtype=np.float32)
        img_b_opponent = np.zeros((img_size, img_size), dtype=np.float32)
        img_free_boxes = np.zeros((img_size, img_size), dtype=np.float32)
        
        # 3) all edges 
        img_edges[::2, 1::2] = 1.0
        img_edges[1::2, ::2] = 1.0
        
        # 4) all boxes
        img_boxes[1::2, 1::2] = 1.0

        # 5) image containing information which lines are drawn (for policy prediction)
        h, v = Board.l_to_h_v(l)
        # horizontals: even rows, odd columns (0-indexing)
        # verticals: odd rows, even columns (0-indexing)
        img_taken_edges[::2, 1::2] = h
        img_taken_edges[1::2, ::2] = v
        img_taken_edges[img_taken_edges == -1.0] = 1.0

        # 6) image indicating boxes captured by player (for value prediction)
        img_b_player[1::2, 1::2] = b
        img_b_player[img_b_player == -1.0] = 0.0

        # 7) image indicating boxes captured by opponent (for value prediction)
        img_b_opponent[1::2, 1::2] = b
        img_b_opponent[img_b_opponent == 1.0] = 0.0
        img_b_opponent[img_b_opponent == -1.0] = 1.0
        
        # 8) boxes that haven't been captured yet
        img_free_boxes[1::2, 1::2] = (b == 0).astype(np.float32)

        feature_planes = np.stack([img_0, img_1, img_edges, img_boxes, img_taken_edges, img_b_player, img_b_opponent, img_free_boxes], axis=0)
        return feature_planes


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''convolution -> N x residual blocks -> policy/value head'''

        x = self.conv_block(x)
        for res_block in self.residual_blocks:
            x = res_block(x)

        p = self.policy_head(x)
        v = self.value_head(x).squeeze()  # one-dimensional output

        return p, v
    
    def p_v(self, l: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Performs a single forward through the neural network. As opposed to forward(), this method ensures that in p
        invalid moves have probability 0 (while still ensuring a probability distribution in p).

        Parameters
        ----------
        l : np.ndarray
            lines vector, assumed to be in its canonical form
        b : np.ndarray
            boxes matrix, assumed to be in its canonical form

        Returns
        -------
        p, v : [np.ndarray, float]
            policy vector p (containing values >= 0 only for valid moves), value v
        """
        valid_moves = np.where(l == 0)[0].tolist()
        assert len(valid_moves) > 0, "No valid move left, model should not be called in this case"

        # model expects ...
        x = self.encode(l, b)
        x = torch.from_numpy(x).to(self.device)  # ... tensor
        x = x.unsqueeze(0)  # ... batch due to batch normalization

        # cpu only necessary when gpu is used
        with torch.no_grad():
            p, v = self.forward(x)
        p = p.squeeze().detach().cpu().numpy()
        v = v.detach().cpu().item()
        
        # softmax
        e = np.exp(p - np.max(p))
        p = e / e.sum()

        # p possibly contains p > 0 for invalid moves -> erase those
        valid = np.zeros(l.squeeze().shape, dtype=np.float32)
        valid[valid_moves] = 1

        p = np.multiply(p, valid)
        
        if np.sum(p) == 0:
            # set probability equally for all valid moves
            p = np.multiply([1] * l.shape[0], valid)

        # normalization to sum 1
        p = p / np.sum(p)

        return p, v


class ConvBlock(nn.Module):
    '''initial convolution -> batch norm -> relu'''

    def __init__(self, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(8, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))

        return x


class ResBlock(nn.Module):
    '''residual block with 2 convolutions'''

    def __init__(self, n_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += x_in
        x = self.relu2(x)

        return x


class PolicyHead(nn.Module):
    '''
    performs a final convolution ->
    flattens the output ->
    linear layer (return num_lines values) ->
    softmax (probabilities for each line)
    '''

    def __init__(self, conv_in_channels, conv_out_channels, kernel_size, stride, padding, fc_in_features, fc_out_features):
        super(PolicyHead, self).__init__()

        self.conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(conv_out_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(
            in_features=fc_in_features,
            out_features=fc_out_features
        )

    def forward(self, x):

        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)

        return x


class ValueHead(nn.Module):
    '''
    performs a final convolution ->
    flattens the output ->
    linear layer (reduce size by 1/2) ->
    linear layer (return 1 value)
    '''

    def __init__(self, conv_in_channels, conv_out_channels, kernel_size, stride, padding, fc_in_features):
        super(ValueHead, self).__init__()

        self.conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(conv_out_channels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(
            in_features=fc_in_features,
            out_features=(fc_in_features//2)
        )
        self.fc2 = nn.Linear(
            in_features=(fc_in_features//2),
            out_features=1
        )


    def forward(self, x):

        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x