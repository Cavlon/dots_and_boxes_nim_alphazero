import numpy as np
import pickle

from pathlib import Path

from game import Board
from nim_utils import transformation_maps, line_groups, grouped_transformations, canonical_transformations, generate_box_checks, distribution_iterator, group_combinations, combinations_iterator, find_pos_ind, next_pos_iterator, apply_map, canonise_pos, mex

def calculate_nim(board, groups, group_sizes, check_A, check_B, canon_trans, trans_maps):

    # used to map bits to apply transformations
    h_flip, v_flip, rot90 = trans_maps

    # maps boards to nimvalues
    dist_dict = np.zeros((1,), dtype=np.int8)

    size = board.SIZE
    num_lines = board.N_LINES

    # save the full board
    path = f"nim/0/"
    for i in range(len(groups)-2):
        path += f"0/"
    path += f"0.bin"
    path_obj = Path(path)

    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path + '_p', 'wb') as handle:
        pickle.dump(dist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Progress from 1 line missing up to all lines missing
    for missing in range(1, num_lines + 1):
        print(missing)

        # find all ways to distribute n missing lines across the groups
        distributions = distribution_iterator(missing, group_sizes)

        for distribution in distributions:

            dist_dict = dict()

            # find all combinations of bits that adhere to the line distribution to form a position
            group_combs = group_combinations(distribution, group_sizes, canon_trans)
            combinations = combinations_iterator(group_combs)

            # calculate the size of the dictionary from all possible combinations
            dict_size = 1
            for comb in group_combs:
                dict_size *= len(comb)
            
            dist_dict = np.zeros((dict_size,), dtype=np.int8)
            dict_ind = 0

            for combination in combinations:
                skip_mex = False

                # read the combination to create the position bitstring
                pos = 0
                shift = 0
                for i in range(len(groups)):
                    pos |= (combination[i] << shift)
                    shift += group_sizes[i]
                
                follower_values = set()

                # find all possible next moves
                follower_moves = next_pos_iterator(pos, groups, group_sizes)

                # create a dynamic distribution for when lines are added
                new_dist = list(distribution)

                prev_group = 0
                for i in range(len(new_dist)):
                    if new_dist[i] > 0:
                        prev_group = i

                new_dist[prev_group] -= 1

                next_group_combs = group_combinations(new_dist, group_sizes, canon_trans)

                # load the dictionary corresponding to the new distribution
                path = f"nim/{missing-1}/"
                for i in range(len(new_dist)-2):
                    path += f"{str(new_dist[i])}/"
                path += f"{new_dist[-2]}.bin"

                with open(path + '_p', 'rb') as handle:
                    saved = pickle.load(handle)

                for next_pos, line, group_ind, pivot_change in follower_moves:

                    if group_ind != prev_group:
                        # update the distribution if a line is added to a new group
                        new_dist[prev_group] += 1
                        new_dist[group_ind] -= 1
                        prev_group = group_ind

                        next_group_combs = group_combinations(new_dist, group_sizes, canon_trans)

                        # load the corresponding dictionary
                        path = f"nim/{missing-1}/"
                        for i in range(len(new_dist)-2):
                            path += f"{str(new_dist[i])}/"
                        path += f"{new_dist[-2]}.bin"

                        with open(path + '_p', 'rb') as handle:
                            saved = pickle.load(handle)
                     
                    capture_box = False

                    # if the line is on an edge then the move can't be loony
                    edge = False

                    # box check: if (pos & check) == 0, the other 3 lines are already 0 (present)
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
                            # this checks if the box check that didn't capture the box still detected 2 lines present
                            # if it did then this move captured a box in a chain and thus entering this position is loony
                            if (res_A > 0 and (res_A & (res_A - 1)) == 0) or (res_B > 0 and (res_B & (res_B - 1)) == 0):
                                dist_dict[dict_ind] = -1
                                skip_mex = True
                                break

                        # if the addition of the line changed the pivot group, it may need to be transformed to return to a canonical position
                        if pivot_change:
                            next_pos = canonise_pos(next_pos, group_sizes[0], canon_trans, h_flip, v_flip, rot90)

                        # find the index of the position in the array
                        next_ind = find_pos_ind(next_pos, next_group_combs, group_sizes)

                        # if a box was captured and it isn't loony then value is the same as the position of the board after the capture
                        dist_dict[dict_ind] = saved[next_ind]
                        skip_mex = True
                        break
                    else:
                        # if the addition of the line changed the pivot group, it may need to be transformed to return to a canonical position
                        if pivot_change:
                            next_pos = canonise_pos(next_pos, group_sizes[0], canon_trans, h_flip, v_flip, rot90)

                        # find the index of the position in the array
                        next_ind = find_pos_ind(next_pos, next_group_combs, group_sizes)
                        follower_values.add(saved[next_ind])
                
                # if no boxes were captured then the value is the mex of all follower boards
                if not skip_mex:
                    dist_dict[dict_ind] = mex(follower_values)
                dict_ind += 1
            
            # save the distribution dictionary
            path = f"nim/{missing}/"
            for i in range(len(distribution)-2):
                path += f"{str(distribution[i])}/"
            path += f"{distribution[-2]}.bin"
            path_obj = Path(path[:-3] + "txt")

            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with path_obj.open("w", encoding="utf-8") as r:
                for value in dist_dict:
                    r.write(f"{value}\n")
            
            with open(path + '_p', 'wb') as handle:
                pickle.dump(dist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    size = 2
    N_LINES = 2 * size * (size + 1)
    print(N_LINES)
    
    board = Board(size)
    trans_maps = transformation_maps(board)

    groups = line_groups(trans_maps, N_LINES)
    print(groups)

    group_sizes = [len(groups[i]) for i in range(len(groups))]
    print(group_sizes)

    g_trans_maps = grouped_transformations(groups, trans_maps, N_LINES)
    print(g_trans_maps)

    with open('canon.txt', 'w') as r:
        canon_map = canonical_transformations(groups, g_trans_maps)
        for key, value in canon_map.items():
            r.write(f"{bin(key)[2:]}: {value}\n")

    with open('checks.txt', 'w') as r:
        check_a, check_b = generate_box_checks(board, groups)
        for i in range(N_LINES):
            r.write(f"{i}:  check a: {bin(check_a[i])[2:] if check_a[i] else check_a[i]}   check b: {bin(check_b[i])[2:] if check_b[i] else check_b[i]}\n")

    calculate_nim(board, groups, group_sizes, check_a, check_b, canon_map, g_trans_maps)

if __name__ == "__main__":
    main()