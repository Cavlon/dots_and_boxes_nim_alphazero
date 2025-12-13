import numpy as np
import random
import math
from typing import Tuple, List

class Board():
    def __init__(self, size: int = 3):
        self.SIZE = size

        # lines
        self.N_LINES = 2 * size * (size + 1)
        self.l = np.zeros((self.N_LINES,), dtype=np.int8)

        # boxes
        self.N_BOXES = size * size
        self.b = np.zeros((self.N_BOXES,), dtype=np.int8)
        
        # maps for connected lines and boxes
        self.line_to_boxes = [self.getConnectedBoxes(i) for i in range(self.N_LINES)]
        self.box_to_lines = [self.getConnectedLines((i, j)) for i in range(size) for j in range(size)]
        
        self.boxes_to_win = math.floor(self.N_BOXES / 2) + 1
    
    def executeMove(self, line: int, player: int):
        assert line < self.N_LINES, "line is out of bounds"
        assert self.l[line] == 0, "line already drawn"
        
        self.l[line] = player
        
        # check whether a new box was captured
        # this is the case when the line belongs to a box (maximum of two boxes) which now has 4 drawn lines
        box_captured = False
        for box in self.line_to_boxes[line]:
            lines = self.box_to_lines[box[0] * self.SIZE + box[1]]
            
            # if all surrounding lines are filled, give the box to the player
            if np.all(self.l[lines] != 0):
                idx = box[0] * self.SIZE + box[1]
                if self.b[idx] == 0:
                    self.b[idx] = player
                    box_captured = True

        return box_captured
    
    def getConnectedBoxes(self, line: int) -> List[Tuple[int, int]]:
        # returns the boxes a line is a part of
        if line < int(self.N_LINES / 2):
            # horizontal line
            i = line // self.SIZE # row of the connected box
            j = line % self.SIZE  # column for both boxes

            if i == 0: # Top line
                return [(i, j)]
            elif i == self.SIZE: # Bottom line
                return [(i - 1, j)]
            else:
                return [(i - 1, j), (i, j)]  # [top box, bottom box]

        else:
            # vertical line
            line = line - int(self.N_LINES / 2)
            j = line // self.SIZE # column of the connected box
            i = line % self.SIZE  # row for both boxes

            if j == 0: # Left line
                return [(i, j)]
            elif j == self.SIZE: # Right line
                return [(i, j - 1)]
            else:
                return [(i, j - 1), (i, j)]  # [left box, right box]
    
    def getConnectedLines(self, box: Tuple[int, int]) -> List[int]:
        # returns the lines surrounding a box
        i = box[0] # Row
        j = box[1] # Column
        
        half = int(self.N_LINES / 2)

        # horizontal lines
        line_top = i * self.SIZE + j  # top line
        line_bottom = (i + 1) * self.SIZE + j  # bottom line

        # vertical lines
        line_left = half + j * self.SIZE + i  # left line
        line_right = half + (j + 1) * self.SIZE + i  # right line

        return [line_top, line_bottom, line_left, line_right]
    
    def getValidMoves(self) -> np.ndarray:
        # returns a boolean array of unfilled lines
        return np.flatnonzero(self.l == 0)
    
    def checkFinished(self):
        # player reached necessary number of captured boxes to win the game
        if ((self.b == 1).sum()) >= self.boxes_to_win:
            return 1

        elif ((self.b == -1).sum()) >= self.boxes_to_win:
            return -1

        elif np.all(self.l != 0):  # no free lines left
            return 0 
        
        else:
            return None
    
    def add_colour(self, text: str, color_code: int) -> str:
        return f"\x1b[{color_code}m{text}\x1b[0m"
    
    def str_horizontal_line(self, line: int, last_column: bool) -> str:

        value = self.l[line]
        color = 31 if value == 1 else 32

        string = "°" + self.add_colour("------", color) if value != 0 else \
            "°  {: >2d}  ".format(line)
        return (string + "°") if last_column else string
    
    def str_vertical_line(self, left_line: int, print_line_number: bool) -> str:

        value = self.l[left_line]
        color = 31 if value == 1 else 32

        if value != 0:
            string = self.add_colour("|", color)

            # color the box when the box right to the line is already captured
            box = self.line_to_boxes(left_line)[-1]
            box_value = self.b[box[0] * self.SIZE + box[1]]
            if box_value == 0:
                return string + "      "
            else:
                color = 31 if box_value == 1 else 32
                return string + self.add_colour("======", color)

        else:
            if print_line_number:
                return "{: >2d}     ".format(left_line)
            else:
                return "       "
    
    def board_string(self) -> str:
        # iterate through boxes from top to bottom, left to right
        string = ""
        for i in range(self.SIZE):

            # 1) use top line
            for j in range(self.SIZE):
                string += self.str_horizontal_line(
                    line=self.box_to_lines((i, j))[0],
                    last_column=(j == self.SIZE - 1)
                )
            string += "\n"

            # 2) use left and right lines
            for repeat in range(3):
                for j in range(self.SIZE):
                    string += self.str_vertical_line(
                        left_line=self.box_to_lines((i, j))[2],
                        print_line_number=(repeat == 1)
                    )

                # last vertical line in a row
                right_line = self.box_to_lines((i, self.SIZE - 1))[3]
                value = self.l[right_line]
                if value != 0:
                    string += self.add_colour("|", 31 if value == 1 else 32)
                else:
                    if repeat == 1:
                        string += f"{right_line}"
                string += "\n"

            # 3) print bottom lines for the last row of boxes
            if i == self.SIZE - 1:
                for j in range(self.SIZE):
                    string += self.str_horizontal_line(
                        line=self.box_to_lines((i, j))[1],
                        last_column=(j == self.SIZE - 1)
                    )
                string += "\n"
        return string
    
    @staticmethod
    def n_lines_to_size(n_lines: int) -> int:
        return int(-0.5 + math.sqrt(4 + 8 * n_lines) / 4)
    
    @staticmethod
    def l_to_h_v(l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert line vector l to (h, v)-matrices representation (containing the horizontal and vertical lines).
        Example (numbers are indices of the line vector l):
            +  0 +  1 +
            6    8    10            [0  1]
        l = +  2 +  3 +    -->  h = [2  3]  and   v = [6  8  10]
            7    9    11            [4  5]            [7  9  11]
            +  4 +  5 +
        """

        n_lines = l.size
        size = Board.n_lines_to_size(n_lines)

        h = np.zeros((size + 1, size), dtype=np.float32)
        v = np.zeros((size, size + 1), dtype=np.float32)

        for line in range(n_lines):
            if line < n_lines / 2:
                # horizontal line
                i = int(line // size)
                j = int(line % size)
                h[i][j] = l[line]

            else:
                # vertical line
                j = int((line - n_lines / 2) // size)
                i = int((line - n_lines / 2) % size)
                v[i][j] = l[line]

        return h, v

    @staticmethod
    def h_v_to_l(h: np.ndarray, v: np.ndarray) -> np.ndarray:

        l = np.concatenate((
            np.matrix.flatten(h, order='C'),  # row-major
            np.matrix.flatten(v, order='F')   # column-major
        ))

        return l

class DotsAndBoxesGame():
    def __init__(self, size: int = 3, starting_player: int = None):
        self.SIZE = size
        self.current_player = (1 if random.random() < 0.5 else -1) if starting_player is None else starting_player
        self.result = None

        self.board = Board(size)
    
    def playMove(self, line: int):
        captured = self.board.executeMove(line, self.current_player)
        
        if captured:
            self.result = self.board.checkFinished()
        else:
            self.current_player *= -1
    
    def getValidMoves(self):
        return self.board.getValidMoves()

    def getCanonicalBoard(self) -> Tuple[np.ndarray, np.ndarray]:
        # returns the board as if the current player were player 1
        canonical_lines = self.current_player * self.board.l
        
        # return the boxes as a 2D matrix
        b2d = self.board.b.reshape((self.SIZE, self.SIZE))
        canonical_boxes = self.current_player * b2d
        return canonical_lines, canonical_boxes

    @staticmethod
    def getSymmetries(l: np.ndarray, b: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # returns a list of rotations and reflections for a board

        # rotations
        h, v = Board.l_to_h_v(l)
        line_rotations = [np.copy(l)]
        
        b = np.copy(b)
        box_rotations = [np.copy(b)]
        
        for i in range(3):
            v, h = np.rot90(h), np.rot90(v)
            line_rotations.append(Board.h_v_to_l(h, v))
            
            box_rotations.append(np.rot90(b))

        # reflections
        line_reflections = []
        box_reflections = []
        for i in range(4):
            h, v = Board.l_to_h_v(line_rotations[i])
            line_reflections.append(Board.h_v_to_l(np.fliplr(h), np.fliplr(v)))
            
            box_reflections.append(np.fliplr(box_rotations[i]))           

        return line_rotations + line_reflections, box_rotations + box_reflections
    
    @staticmethod
    def getLineSymmetries(l: np.ndarray) -> List[np.ndarray]:

        # rotations
        h, v = Board.l_to_h_v(l)
        line_rotations = [np.copy(l)]
        
        for i in range(3):
            v, h = np.rot90(h), np.rot90(v)
            line_rotations.append(Board.h_v_to_l(h, v))

        # reflections
        line_reflections = []
        for i in range(4):
            h, v = Board.l_to_h_v(line_rotations[i])
            line_reflections.append(Board.h_v_to_l(np.fliplr(h), np.fliplr(v)))      

        return line_rotations + line_reflections
    
    @staticmethod
    def clone(s):
        new = DotsAndBoxesGame(s.SIZE, s.current_player)
        new.result = s.result
        new.board.l = s.board.l.copy()
        new.board.b = s.board.b.copy()
        return new

    def display(self):
        print(self.board.board_string())
    
    def status(self):
        print(f"Player 1: {(self.board.b == 1).sum()}    Player 2: {(self.board.b == -1).sum()}")
        print(f"Player {2 if self.current_player == -1 else 1}'s Turn")