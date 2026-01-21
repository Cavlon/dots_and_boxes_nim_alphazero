from game import Board
from nim_utils import transformation_maps, line_groups, grouped_transformations, canonical_transformations, generate_box_checks, distribution_iterator, group_combinations_iterator, next_pos_iterator, apply_map, canonise_pos, mex

def calculate_nim(board, groups, group_map, group_sizes, check_A, check_B, canon_trans, trans_maps):

    # used to map bits to apply transformations
    h_flip, v_flip, rot90 = trans_maps

    # maps boards to nimvalues
    saved = {0: 0} # full board has value 0
    curr_saved = dict()

    size = board.SIZE
    num_lines = board.N_LINES
    
    # Progress from 1 line missing up to all lines missing
    for missing in range(1, num_lines + 1):
        print(missing)

        # find all ways to distribute n missing lines across the groups
        distributions = distribution_iterator(missing, group_sizes)

        for distribution in distributions:

            # find all combinations of bits that adhere to the line distribution to form a position
            group_combinations = group_combinations_iterator(distribution, group_sizes, canon_trans)

            for combination in group_combinations:
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

                for next_pos, line, pivot_change in follower_moves:
                     
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
                                curr_saved[pos] = -1
                                skip_mex = True
                                break

                        # if the addition of the line changed the pivot group, it may need to be transformed to return to a canonical position
                        if pivot_change:
                            next_pos = canonise_pos(next_pos, group_sizes[0], canon_trans, h_flip, v_flip, rot90)

                        # if a box was captured and it isn't loony then value is the same as the position of the board after the capture
                        curr_saved[pos] = saved[next_pos]
                        skip_mex = True
                        break
                    else:
                        # if the addition of the line changed the pivot group, it may need to be transformed to return to a canonical position
                        if pivot_change:
                            next_pos = canonise_pos(next_pos, group_sizes[0], canon_trans, h_flip, v_flip, rot90)

                        follower_values.add(saved[next_pos])
                
                # if no boxes were captured then the value is the mex of all follower boards
                if not skip_mex:
                    curr_saved[pos] = mex(follower_values)
        
        # write the dictionary to the disk
        with open(f'nim/nim_{missing}.txt', 'w') as r:
            for key, value in curr_saved.items():
                r.write(f"{bin(key)[2:]}: {value}\n")
        
        # remove the old dictionary from memory
        saved = curr_saved
        curr_saved = dict()

def main():
    size = 2
    N_LINES = 2 * size * (size + 1)
    print(N_LINES)
    
    board = Board(size)
    trans_maps = transformation_maps(board)

    groups, group_map = line_groups(trans_maps, N_LINES)
    print(groups)
    print(group_map)

    group_sizes = [len(groups[i]) for i in range(len(groups))]
    print(group_sizes)

    g_trans_maps = grouped_transformations(groups, trans_maps, N_LINES)
    print(g_trans_maps)

    pivot_group = 0

    with open('canon.txt', 'w') as r:
        canon_map = canonical_transformations(groups, pivot_group, g_trans_maps)
        for key, value in canon_map.items():
            r.write(f"{bin(key)[2:]}: {value}\n")

    with open('checks.txt', 'w') as r:
        check_a, check_b = generate_box_checks(board, groups, group_map)
        for i in range(N_LINES):
            r.write(f"{i}:  check a: {bin(check_a[i])[2:] if check_a[i] else check_a[i]}   check b: {bin(check_b[i])[2:] if check_b[i] else check_b[i]}\n")

    calculate_nim(board, groups, group_map, group_sizes, check_a, check_b, canon_map, g_trans_maps)

if __name__ == "__main__":
    main()