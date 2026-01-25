import numpy as np
import pickle
import cProfile
import pstats

from pathlib import Path

from game import Board
from nim_utils import *

GLOBAL_GROUPS = None
GLOBAL_GROUP_SIZES = None
GLOBAL_GROUP_COUNT = None
GLOBAL_GROUP_SHIFTS = []
GLOBAL_B2G_MAP = dict()
GLOBAL_B2L_MAP = None

GLOBAL_CHECK_A = None
GLOBAL_CHECK_B = None

GLOBAL_CANON_TRANS = None

GLOBAL_HFLIP = None
GLOBAL_VFLIP = None
GLOBAL_ROT90 = None

def distribution_analyse(distribution, missing):
    # find all combinations of bits that adhere to the line distribution to form a position
    group_combs, comb_map = group_combinations(distribution, GLOBAL_GROUP_SIZES, GLOBAL_CANON_TRANS)
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
        for i in range(GLOBAL_GROUP_COUNT):
            pos |= (combination[i] << GLOBAL_GROUP_SHIFTS[i])
        
        follower_values = set()

        # find all possible next moves
        follower_moves = next_pos_iterator(pos, GLOBAL_GROUPS, GLOBAL_GROUP_SIZES, GLOBAL_GROUP_SHIFTS, GLOBAL_B2G_MAP, GLOBAL_B2L_MAP)

        # create a dynamic distribution for when lines are added
        new_dist = list(distribution)

        prev_group = 0
        for i in range(GLOBAL_GROUP_COUNT):
            if new_dist[i] > 0:
                prev_group = i
                break

        new_dist[prev_group] -= 1

        next_group_combs = [group[:] for group in group_combs]
        next_comb_map = [dict(c_map) for c_map in comb_map]
        prev_combs, prev_map = next_group_combs[prev_group], next_comb_map[prev_group] 
        next_group_combs[prev_group], next_comb_map[prev_group] = update_group_combinations(prev_group, new_dist, GLOBAL_GROUP_SIZES, GLOBAL_GROUP_COUNT, GLOBAL_CANON_TRANS)

        ind_multipliers = []
        mult = 1
        for i in range(GLOBAL_GROUP_COUNT):
            ind_multipliers.append(mult)
            mult *= len(next_group_combs[i])

        # load the dictionary corresponding to the new distribution
        path = f"nim/{missing-1}/"
        for i in range(GLOBAL_GROUP_COUNT-2):
            path += f"{str(new_dist[i])}/"
        path += f"{new_dist[-2]}.bin"

        with open(path + '_p', 'rb') as handle:
            saved = pickle.load(handle)
        
        # print(f"dist: {distribution}    new_dist:{new_dist}     new_combs:{next_group_combs}")

        for next_pos, line, group_ind, pivot_change in follower_moves:

            if group_ind != prev_group:
                # print('update')
                # update the distribution if a line is added to a new group
                new_dist[prev_group] += 1
                new_dist[group_ind] -= 1

                next_group_combs[prev_group], next_comb_map[prev_group] = prev_combs, prev_map
                prev_combs, prev_map = next_group_combs[group_ind], next_comb_map[group_ind]

                next_group_combs[group_ind], next_comb_map[group_ind] = update_group_combinations(group_ind, new_dist, GLOBAL_GROUP_SIZES, GLOBAL_GROUP_COUNT, GLOBAL_CANON_TRANS)

                prev_group = group_ind

                ind_multipliers = []
                mult = 1
                for i in range(GLOBAL_GROUP_COUNT):
                    ind_multipliers.append(mult)
                    mult *= len(next_group_combs[i])

                # load the corresponding dictionary
                path = f"nim/{missing-1}/"
                for i in range(GLOBAL_GROUP_COUNT-2):
                    path += f"{str(new_dist[i])}/"
                path += f"{new_dist[-2]}.bin"

                with open(path + '_p', 'rb') as handle:
                    saved = pickle.load(handle)
            
            # print(new_dist)
            capture_box = False

            # if the line is on an edge then the move can't be loony
            edge = False

            # box check: if (pos & check) == 0, the other 3 lines are already 0 (present)
            if GLOBAL_CHECK_A[line]:
                res_A = pos & GLOBAL_CHECK_A[line]
                if res_A == 0:
                    capture_box = True
            else:
                edge = True
            
            if GLOBAL_CHECK_B[line]:
                res_B = pos & GLOBAL_CHECK_B[line]
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
                    next_pos = canonise_pos(next_pos, GLOBAL_GROUP_SIZES[0], GLOBAL_CANON_TRANS, GLOBAL_HFLIP, GLOBAL_VFLIP, GLOBAL_ROT90)

                # find the index of the position in the array
                next_ind = find_pos_ind(next_pos, next_group_combs, GLOBAL_GROUP_SIZES, ind_multipliers, next_comb_map, GLOBAL_GROUP_SHIFTS)

                # if a box was captured and it isn't loony then value is the same as the position of the board after the capture
                dist_dict[dict_ind] = saved[next_ind]
                skip_mex = True
                break
            else:
                # if the addition of the line changed the pivot group, it may need to be transformed to return to a canonical position
                if pivot_change:
                    next_pos = canonise_pos(next_pos, GLOBAL_GROUP_SIZES[0], GLOBAL_CANON_TRANS, GLOBAL_HFLIP, GLOBAL_VFLIP, GLOBAL_ROT90)

                # find the index of the position in the array
                next_ind = find_pos_ind(next_pos, next_group_combs, GLOBAL_GROUP_SIZES, ind_multipliers, next_comb_map, GLOBAL_GROUP_SHIFTS)
                follower_values.add(saved[next_ind])
        
        # if no boxes were captured then the value is the mex of all follower boards
        if not skip_mex:
            dist_dict[dict_ind] = mex(follower_values)
        dict_ind += 1
    
    # save the distribution dictionary
    path = f"nim/{missing}/"
    for i in range(GLOBAL_GROUP_COUNT-2):
        path += f"{str(distribution[i])}/"
    path += f"{distribution[-2]}.bin"
    path_obj = Path(path[:-3] + "txt")

    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with path_obj.open("w", encoding="utf-8") as r:
        for value in dist_dict:
            r.write(f"{value}\n")
    
    with open(path + '_p', 'wb') as handle:
        pickle.dump(dist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def nim_analyse(num_lines):
    # maps boards to nimvalues
    dist_dict = np.zeros((1,), dtype=np.int8)

    # save the full board
    path = f"nim/0/"
    for i in range(GLOBAL_GROUP_COUNT-2):
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
        distributions = distribution_iterator(missing, GLOBAL_GROUP_SIZES, GLOBAL_GROUP_COUNT, 0, sum(GLOBAL_GROUP_SIZES))

        for distribution in distributions:
            distribution_analyse(distribution, missing)

def main():
    global GLOBAL_GROUPS
    global GLOBAL_GROUP_SIZES
    global GLOBAL_GROUP_COUNT

    global GLOBAL_GROUP_SHIFTS
    global GLOBAL_B2G_MAP
    global GLOBAL_B2L_MAP

    global GLOBAL_HFLIP
    global GLOBAL_VFLIP
    global GLOBAL_ROT90

    global GLOBAL_CANON_TRANS

    global GLOBAL_CHECK_A
    global GLOBAL_CHECK_B

    size = 3
    num_lines = 2 * size * (size + 1)
    print(num_lines)
    
    board = Board(size)
    trans_maps = transformation_maps(size, num_lines)

    GLOBAL_GROUPS = line_groups(trans_maps, num_lines)
    GLOBAL_GROUP_COUNT = len(GLOBAL_GROUPS)
    print(GLOBAL_GROUPS)

    GLOBAL_B2L_MAP = [ind for group in GLOBAL_GROUPS for ind in group]

    GLOBAL_GROUP_SIZES = [len(GLOBAL_GROUPS[i]) for i in range(GLOBAL_GROUP_COUNT)]
    print(GLOBAL_GROUP_SIZES)

    shift = 0
    for size in GLOBAL_GROUP_SIZES:
        GLOBAL_GROUP_SHIFTS.append(shift)
        shift += size
    
    group_ind = 0
    for i in range(num_lines):
        if group_ind < GLOBAL_GROUP_COUNT-1 and i == GLOBAL_GROUP_SHIFTS[group_ind+1]:
            group_ind += 1
        GLOBAL_B2G_MAP[i] = group_ind

    GLOBAL_HFLIP, GLOBAL_VFLIP, GLOBAL_ROT90 = grouped_transformations(GLOBAL_GROUPS, trans_maps, num_lines)
    print(GLOBAL_HFLIP, GLOBAL_VFLIP, GLOBAL_ROT90)

    GLOBAL_CANON_TRANS = canonical_transformations(GLOBAL_GROUPS, GLOBAL_HFLIP, GLOBAL_VFLIP, GLOBAL_ROT90)
    GLOBAL_CHECK_A, GLOBAL_CHECK_B = generate_box_checks(board, GLOBAL_GROUPS, GLOBAL_GROUP_SHIFTS)

    profiler = cProfile.Profile()
    profiler.enable()

    nim_analyse(num_lines)

    profiler.disable()

    with open("profile_results2.log", "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
        stats.print_stats()

if __name__ == "__main__":
    main()