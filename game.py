import numpy as np
import math
from typing import Tuple, List

class Board():
    def __init__(self, size: int = 3):
        self.SIZE = size

        # lines
        self.N_LINES = 2 * size * (size + 1)
        self.l = np.zeros((self.N_LINES,), dtype=np.float32)

        # boxes
        self.N_BOXES = size * size
        self.b = np.zeros((size, size))
    
    def executeMove(self, line: int, player: int):
        assert line < self.N_LINES, "line is out of bounds"
        assert self.l[line] == 0, "line already drawn"
        
        self.l[line] = player
        
        # check whether a new box was captured
        # this is the case when the line belongs to a box (maximum of two boxes) which now has 4 drawn lines
        box_captured = False
        for box in self.getConnectedBoxes(line):
            lines = self.getConnectedLines(box)

            if len([self.l[line] for line in lines if self.l[line] != 0]) == 4: # If all connected lines are != 0 (taken)
                assert self.b[box[0]][box[1]] == 0, "box already captured"
                self.b[box[0]][box[1]] = player
                box_captured = True

        return box_captured
    
    def getConnectedBoxes(self, line: int) -> List[Tuple[int, int]]:
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
        i = box[0] # Row
        j = box[1] # Column

        # horizontal lines
        line_top = i * self.SIZE + j  # top line
        line_bottom = (i + 1) * self.SIZE + j  # bottom line

        # vertical lines
        line_left = int(self.N_LINES / 2) + j * self.SIZE + i  # left line
        line_right = int(self.N_LINES / 2) + (j + 1) * self.SIZE + i  # right line

        return [line_top, line_bottom, line_left, line_right]
    
    def getValidMoves(self) -> List[int]:
        return np.where(self.l == 0)[0].tolist()
    
    def checkFinished(self):
        # player reached necessary number of captured boxes to win the game
        boxes_to_win = math.floor(self.N_BOXES / 2) + 1

        if ((self.b == 1).sum()) >= boxes_to_win:
            return 1

        elif ((self.b == -1).sum()) >= boxes_to_win:
            return -1

        else:
            return None  # not finished
    
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
            box = self.getConnectedBoxes(left_line)[-1]
            box_value = self.b[box[0], box[1]]
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
                    line=self.getConnectedLines((i, j))[0],
                    last_column=(j == self.SIZE - 1)
                )
            string += "\n"

            # 2) use left and right lines
            for repeat in range(3):
                for j in range(self.SIZE):
                    string += self.str_vertical_line(
                        left_line=self.getConnectedLines((i, j))[2],
                        print_line_number=(repeat == 1)
                    )

                # last vertical line in a row
                right_line = self.getConnectedLines((i, self.SIZE - 1))[3]
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
                        line=self.getConnectedLines((i, j))[1],
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
    def __init__(self, size: int = 3):
        self.SIZE = size
        self.current_player = 1
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
        canonical_lines = self.current_player * self.board.l
        canonical_lines[canonical_lines == 0.] = 0.  # -0.0 to 0.0
        
        canonical_boxes = self.current_player * self.board.b
        canonical_boxes[canonical_boxes == 0.] = 0.  # -0.0 to 0.0

        return canonical_lines, canonical_boxes

    @staticmethod
    def getSymmetries(l: np.ndarray, b: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:

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

    def display(self):
        print(self.board.board_string())
    
    def status(self):
        print(f"Player 1: {(self.board.b == 1).sum()}    Player 2: {(self.board.b == -1).sum()}")
        print(f"Player {2 if self.current_player == -1 else 1}'s Turn")