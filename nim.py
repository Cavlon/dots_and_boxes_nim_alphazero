import numpy as np
import pickle
import cProfile
import pstats
import multiprocessing

from functools import partial
from pathlib import Path

from game import Board
from nim_utils import *

def distribution_analyse(
    distribution, missing, 
    groups, group_sizes, group_count, group_shifts,
    b2g_map, b2l_map,
    check_a, check_b,
    canon_trans,
    h_flip, v_flip, rot_90,
    pivot_c_map, corner_c_map, c_8_map, c_4_map, c_map
):

    target_path = f"nim/{missing}/{'/'.join(map(str, distribution[:-1]))}.bin"
    target_path = Path(target_path)

    # skip this distribution if it has already been processed
    if target_path.exists():
        return
    
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # find all combinations of bits that adhere to the line distribution to form a position
    group_combs = group_combinations(distribution, group_sizes, canon_trans)
    combinations = combinations_iterator(group_combs)

    # calculate the size of the dictionary from all possible combinations
    dict_size = 1
    for comb in group_combs:
        dict_size *= len(comb)
    
    dict_size = (dict_size >> 1) + 1 if dict_size % 2 != 0 else dict_size >> 1
    
    dist_dict = np.zeros((dict_size,), dtype=np.uint8)
    dict_ind = 0

    dict_val = 0
    full_byte = True

    for combination in combinations:
        skip_mex = False
        pos_val = 0

        # read the combination to create the position bitstring
        pos = 0
        for i in range(group_count):
            pos |= (combination[i] << group_shifts[i])
        
        follower_values = set()

        # find all possible next moves
        follower_moves = next_pos_iterator(pos, groups, group_sizes, group_shifts, b2g_map, b2l_map)

        # create a dynamic distribution for when lines are added
        new_dist = list(distribution)

        prev_group = 0
        for i in range(group_count):
            if new_dist[i] > 0:
                prev_group = i
                break

        new_dist[prev_group] -= 1

        ind_multipliers = [1]
        mult = pivot_c_map[new_dist[0]]
        for i in range(1, group_count-1):
            ind_multipliers.append(mult)

            if group_sizes[i] == 8:      
                mult *= c_8_map[new_dist[i]]
            else:
                mult *= c_4_map[new_dist[i]]
        ind_multipliers.append(mult)

        # load the dictionary corresponding to the new distribution
        path = f"nim/{missing-1}/{'/'.join(map(str, new_dist[:-1]))}.bin"

        saved = np.memmap(path, dtype='uint8', mode='r')

        for next_pos, line, group_ind, pivot_change in follower_moves:

            if group_ind > prev_group:
                # update the distribution if a line is added to a new group
                new_dist[prev_group] += 1
                new_dist[group_ind] -= 1

                prev_group = group_ind

                ind_multipliers = [1]
                mult = pivot_c_map[new_dist[0]]
                for i in range(1, group_count-1):
                    ind_multipliers.append(mult)

                    if group_sizes[i] == 8:      
                        mult *= c_8_map[new_dist[i]]
                    else:
                        mult *= c_4_map[new_dist[i]]
                ind_multipliers.append(mult)
                
                path = f"nim/{missing-1}/{'/'.join(map(str, new_dist[:-1]))}.bin"

                saved = np.memmap(path, dtype='uint8', mode='r')
            
            # box check: if (pos & check) == 0, the other 3 lines are already 0 (present)            
            res_A = pos & check_a[line] if check_a[line] else None
            res_B = pos & check_b[line] if check_b[line] else None
            
            capture_box = (res_A == 0) or (res_B == 0)
            # if the line is on an edge then the move can't be loony
            edge = (res_A is None) or (res_B is None)

            if capture_box:
                if not edge:
                    # this checks if the box check that didn't capture the box still detected 2 lines present
                    # if it did then this move captured a box in a chain and thus entering this position is loony
                    if (res_A > 0 and (res_A & (res_A - 1)) == 0) or (res_B > 0 and (res_B & (res_B - 1)) == 0):
                        pos_val = 15
                        skip_mex = True
                        break

                # if the addition of the line changed the pivot group, it may need to be transformed to return to a canonical position
                if pivot_change:
                    next_pos = canonise_pos(next_pos, group_sizes[0], canon_trans, h_flip, v_flip, rot_90)

                # find the index of the position in the array
                next_ind = find_pos_ind(next_pos, group_sizes, ind_multipliers, c_map, group_shifts)
                saved_byte = saved[next_ind >> 1]

                # if a box was captured and it isn't loony then value is the same as the position of the board after the capture
                # extract the relevant 4 bits from the byte
                pos_val = saved_byte & 15 if next_ind % 2 == 0 else saved_byte >> 4
                skip_mex = True
                break
            else:
                # if the addition of the line changed the pivot group, it may need to be transformed to return to a canonical position
                if pivot_change:
                    next_pos = canonise_pos(next_pos, group_sizes[0], canon_trans, h_flip, v_flip, rot_90)

                # find the index of the position in the array
                next_ind = find_pos_ind(next_pos, group_sizes, ind_multipliers, c_map, group_shifts)
                saved_byte = saved[next_ind >> 1]

                # extract the relevant 4 bits from the byte
                val = saved_byte & 15 if next_ind % 2 == 0 else saved_byte >> 4
                follower_values.add(val)
        
        # if no boxes were captured then the value is the mex of all follower boards
        if not skip_mex:
            pos_val = mex(follower_values)
        
        # if there are enough bits to form a byte then write it to the file
        if full_byte:
            dict_val = pos_val
        else:
            dict_val |= (pos_val << 4)
            dist_dict[dict_ind] = dict_val
            dict_ind += 1

        full_byte = not full_byte
    
    # write any leftover bits
    if not full_byte:
        dist_dict[dict_ind] = dict_val

    # save the distribution dictionary
    # text_path = target_dir / f"{distribution[-2]}.txt"
    # with text_path.open("w", encoding="utf-8") as r:
    #     for value in dist_dict:
    #         r.write(f"{value & 15}\n{value >> 4}\n")
    
    dist_dict.tofile(target_path)

def nim_analyse(
    num_lines, 
    groups, group_sizes, group_count, group_shifts,
    b2g_map, b2l_map,
    check_a, check_b,
    canon_trans,
    h_flip, v_flip, rot_90,
    pivot_c_map, corner_c_map, c_8_map, c_4_map, c_map
):
    # maps boards to nimvalues
    dist_dict = np.zeros((1,), dtype=np.uint8)

    # save the full board
    path = f"nim/0/"
    for i in range(group_count-2):
        path += f"0/"
    path += f"0.bin"
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    dist_dict.tofile(path)
    
    # use as many cores as possible to compute distibutions in parallel
    with multiprocessing.Pool() as pool:
        # Progress from 1 line missing up to all lines missing
        for missing in range(1, num_lines + 1):
            print(missing)

            # find all ways to distribute n missing lines across the groups
            distributions = distribution_iterator(missing, group_sizes, group_count, 0, num_lines)

            # fix all static arguments
            worker = partial(
                distribution_analyse, missing=missing, 
                groups=groups, group_sizes=group_sizes, group_count=group_count, group_shifts=group_shifts,
                b2g_map=b2g_map, b2l_map=b2l_map,
                check_a=check_a, check_b=check_b,
                canon_trans=canon_trans,
                h_flip=h_flip, v_flip=v_flip, rot_90=rot_90,
                pivot_c_map=pivot_c_map, corner_c_map=corner_c_map, c_8_map=c_8_map, c_4_map=c_4_map, c_map=c_map
            )

            pool.map(worker, distributions)

def main():
    size = 2
    num_lines = 2 * size * (size + 1)
    print(num_lines)
    
    board = Board(size)
    trans_maps = transformation_maps(size, num_lines)

    groups = line_groups(trans_maps, num_lines)
    group_count = len(groups)
    print(groups)

    b2l_map = [ind for group in groups for ind in group]
    b2l_map = np.array(b2l_map, dtype=np.uint8)

    group_sizes = [len(groups[i]) for i in range(group_count)]
    print(group_sizes)

    shift = 0
    group_shifts = []
    for size in group_sizes:
        group_shifts.append(shift)
        shift += size
    
    group_ind = 0
    b2g_map = np.zeros((num_lines,), dtype=np.uint8)
    for i in range(num_lines):
        if group_ind < group_count-1 and i == group_shifts[group_ind+1]:
            group_ind += 1
        b2g_map[i] = group_ind

    h_flip, v_flip, rot_90 = grouped_transformations(groups, trans_maps, num_lines)
    print(h_flip, v_flip, rot_90)

    canon_trans = canonical_transformations(groups, h_flip, v_flip, rot_90)
    check_a, check_b = generate_box_checks(board, groups, group_shifts)

    pivot_map, pivot_c_map = generate_pivot_comb_map(group_sizes[0], canon_trans)
    corner_map, corner_c_map = generate_corner_comb_map()
    map_8, c_8_map = generate_comb_map(8)
    map_4, c_4_map = generate_comb_map(4)

    c_map = [pivot_map]
    for i in range(1, group_count-1):
        if group_sizes[i] == 8:
         c_map.append(map_8)
        else:
         c_map.append(map_4)
    c_map.append(corner_map)

    # profiler = cProfile.Profile()
    # profiler.enable()

    nim_analyse(
        num_lines,
        groups, group_sizes, group_count, group_shifts,
        b2g_map, b2l_map,
        check_a, check_b,
        canon_trans,
        h_flip, v_flip, rot_90,
        pivot_c_map, corner_c_map, c_8_map, c_4_map, c_map
    )

    # profiler.disable()

    # with open("profile_results.log", "w") as f:
    #     stats = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
    #     stats.print_stats()

if __name__ == "__main__":
    main()