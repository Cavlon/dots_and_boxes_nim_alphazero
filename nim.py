import itertools
import numpy as np

from game import Board

def mex(values):
    mex_val = 0
    while mex_val in values:
        mex_val += 1
    return mex_val

def transformation_maps(board: Board):
    num_lines = board.N_LINES
    size = board.SIZE
    half = num_lines // 2

    h_flip = [0] * num_lines
    v_flip = [0] * num_lines
    rot_90 = [0] * num_lines

    for row in range(size + 1):
        for col in range(size + 1):
            # Transform Horizontal Lines
            if col < size:
                ind = row * size + col
                h_flip[ind] = row * size + (size - col - 1)
                v_flip[ind] = (size - row) * size + col

                rot_90[ind] = half + (size-row) * size + col
            
            # Transform Vertical Lines
            if row < size:
                ind = half + col * size + row
                h_flip[ind] = half + (size - col) * size + row
                v_flip[ind] = half + col * size + (size - row - 1)
  
                rot_90[ind] = col * size + (size - row - 1)
                
    return h_flip, v_flip, rot_90

def line_groups(maps, num_lines):
    h_flip, v_flip, rot90 = maps

    groups = []
    processed = set()
    group_map = dict()
    group_ind = -1
    
    for i in range(num_lines):
        if i not in processed:
            group_ind += 1
            ind = i
            visited = set()

            for j in range(2):
                h_ind = h_flip[ind]
                v_ind = v_flip[ind]

                visited.add(ind)
                visited.add(h_ind)
                visited.add(v_ind)

                processed.add(ind)
                processed.add(h_ind)
                processed.add(v_ind)

                group_map[ind] = group_ind
                group_map[h_ind] = group_ind
                group_map[v_ind] = group_ind

                ind = rot90[ind]
            
            visited.add(ind)
            visited.add(rot90[ind])

            processed.add(ind)
            processed.add(rot90[ind])

            group_map[ind] = group_ind
            group_map[rot90[ind]] = group_ind

            groups.append(sorted(list(visited)))
    
    return groups, group_map

def grouped_transformations(groups, group_map, maps, num_lines):

    h_flip, v_flip, rot90 = maps

    g_h_flip = [0] * num_lines
    g_v_flip = [0] * num_lines
    g_rot_90 = [0] * num_lines

    for i in range(num_lines):
        group_ind = group_map[i]
        
        trans_ind = h_flip[i]
        g_h_flip[i] = groups[group_ind].index(trans_ind)

        trans_ind = v_flip[i]
        g_v_flip[i] = groups[group_ind].index(trans_ind)

        trans_ind = rot90[i]
        g_rot_90[i] = groups[group_ind].index(trans_ind)
    
    return g_h_flip, g_v_flip, g_rot_90

def apply_group_map(state, trans_map, group_len):
    new_s = 0
    for bit in range(group_len):
        if (state >> bit) & 1:
            new_s |= (1 << trans_map[groups[1][bit]])
    return new_s

def canonical_transformations(groups, maps):
    h_flip, v_flip, rot90 = maps

    canon_trans = {0:0}
    
    group_len = len(groups[1])

    for i in range(1, group_len+1):
        canon = (1 << i) - 1
        canon_trans[i] = 0

        h_pos = apply_group_map(canon, h_flip, group_len)
        v_pos = apply_group_map(canon, v_flip, group_len)
        rot_pos = apply_group_map(canon, rot90, group_len)
        hrot_pos = apply_group_map(rot_pos, h_flip, group_len)
        vrot_pos = apply_group_map(rot_pos, v_flip, group_len)
        rot2_pos = apply_group_map(rot_pos, rot90, group_len)
        rot3_pos = apply_group_map(rot2_pos, rot90, group_len)

        canon_trans[h_pos] = 1
        canon_trans[v_pos] = 2
        canon_trans[rot_pos] = 3
        canon_trans[hrot_pos] = 4
        canon_trans[vrot_pos] = 5
        canon_trans[rot2_pos] = 6
        canon_trans[rot3_pos] = 7

        # print(f"canon: {bin(canon)[2:]}     h_flip: {bin(h_pos)[2:]}    v_flip: {bin(v_pos)[2:]}    rot: {bin(rot_pos)[2:]}")
    
    return canon_trans        

def generate_box_checks(board: Board):
    
    # Initialize with all bits set (1 means line is missing)
    n_lines = board.N_LINES
    size = board.SIZE
    check_A = [None] * n_lines
    check_B = [None] * n_lines

    # iterate through each box
    for row in range(size):
        for col in range(size):
            # [top, bottom, left, right] lines of a box
            lines = board.box_to_lines[row * size + col]
            
            # iterate through each line
            for i in range(4):
                current_line = lines[i]
                mask = 0

                # add the other 3 lines to the mask
                for j in range(4):
                    if i != j:
                        mask |= (1 << lines[j])
                
                # a is if box is top or left, b is if box is bottom or right
                if i % 2 == 0:
                    check_B[current_line] = mask
                else:
                    check_A[current_line] = mask
                    
    return check_A, check_B

def calculate_nim(board, check_A, check_B):
    saved = {0: 0} # full board has value 0

    size = board.SIZE
    num_lines = board.N_LINES

    indices = range(num_lines)
    
    # Progress from 1 line missing up to all lines missing
    for missing in range(1, num_lines + 1):
        print(missing)
        for combination in itertools.combinations(indices, missing):
            skip_mex = False

            # 1 = missing, 0 = present
            pos = 0
            for index in combination:
                pos |= (1 << index)
            
            follower_values = []
            for line in combination: # Only try lines that are missing
                next_pos = pos ^ (1 << line)
                
                # Box check: if (state & mask) == 0, the other 3 lines were already 0
                capture_box = False
                edge = False

                if check_A[line]:
                    res_A = pos & check_A[line]
                    if res_A == 0:
                        capture_box = True
                else:
                    edge = True
                
                if check_B[line]:
                    res_B = pos & check_B[line]
                    if res_B == 0:
                        capture_box = True
                else:
                    edge = True

                        
                
                if capture_box:
                    if not edge:
                        if (res_A > 0 and (res_A & (res_A - 1)) == 0) or (res_B > 0 and (res_B & (res_B - 1)) == 0):
                            saved[pos] = -1
                            skip_mex = True
                            break

                    # Extra turn: value is the nim-value of the resulting state
                    saved[pos] = saved[next_pos]
                    skip_mex = True
                    break
                else:
                    # Normal move: standard nim-value logic
                    follower_values.append(saved[next_pos])
            
            if not skip_mex:
                saved[pos] = mex(follower_values)
            
    return saved

def main():
    size = 2
    N_LINES = 2 * size * (size + 1)
    print(N_LINES)
    
    board = Board(size)
    trans_maps = transformation_maps(board)

    groups, group_map = line_groups(trans_maps, N_LINES)
    print(groups)
    print(group_map)

    g_trans_maps = grouped_transformations(groups, group_map, trans_maps, N_LINES)
    print(g_trans_maps)

    canonical_transformations(groups, g_trans_maps)

    # with open('checks.txt', 'w') as r:
    #     a, b = generate_box_checks(board)
    #     for i in range(N_LINES):
    #         r.write(f"{i}:  check a: {bin(a[i]) if a[i] else a[i]}   check b: {bin(b[i]) if b[i] else b[i]}\n")

    # with open('nim.txt', 'w') as r:
    #     saved = calculate_nim(board, a, b)
    #     for key, value in saved.items():
    #         r.write(f"{bin(key)[2:]}: {value}\n")

if __name__ == "__main__":
    main()