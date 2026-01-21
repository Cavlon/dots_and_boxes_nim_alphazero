import itertools
import numpy as np

from game import Board

def mex(values):
    '''
    Finds the minimum excluded value from a set of values
    '''
    mex_val = 0
    while mex_val in values:
        mex_val += 1
    return mex_val

def transformation_maps(board: Board):
    '''
    Constructs transformation maps for horizontal and vertical flips, and 90 degree clockwise rotation
    The map is for a bitstring ordered by index
    '''
    num_lines = board.N_LINES
    size = board.SIZE
    half = num_lines // 2

    h_flip = [0] * num_lines
    v_flip = [0] * num_lines
    rot_90 = [0] * num_lines

    # iterate though each line via column and row
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
    '''
    Splits line indices into groups based on symmetries
    Lines that map to each other are part of the same group
    '''
    h_flip, v_flip, rot90 = maps

    groups = []
    processed = set()
    group_map = dict()  # maps a line index to its corresponding group
    group_ind = -1
    
    for i in range(num_lines):
        # if this line hasn't been added to a group yet, create a new group
        if i not in processed:
            group_ind += 1
            ind = i
            visited = set()

            # perform every transformation and add the visited lines to the group
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

def grouped_transformations(groups, maps, num_lines):
    '''
    Constructs transformation maps for the reordered bitstring based on groups
    '''
    h_flip, v_flip, rot90 = maps

    g_h_flip = [0] * num_lines
    g_v_flip = [0] * num_lines
    g_rot_90 = [0] * num_lines

    # maps line indices to their bit positions in the grouped bitstring
    flattened = [ind for group in groups for ind in group]

    # find the new transformed indices for each line in the new grouped bitstring
    for i in range(len(flattened)):
        trans_ind = h_flip[flattened[i]]
        g_h_flip[i] = flattened.index(trans_ind)

        trans_ind = v_flip[flattened[i]]
        g_v_flip[i] = flattened.index(trans_ind)

        trans_ind = rot90[flattened[i]]
        g_rot_90[i] = flattened.index(trans_ind)
    
    return g_h_flip, g_v_flip, g_rot_90

def apply_map(pos, trans_map):
    '''
    Applies a transformation map on a bitstring
    '''
    new_pos = 0
    for bit in range(pos.bit_length()):
        if (pos >> bit) & 1:
            new_pos |= (1 << trans_map[bit])
    return new_pos

def canonical_transformations(groups, pivot_ind, maps):
    '''
    Iterates through each possible combination of group bits in the pivot and applies every transformation to them
    The variant with the lowest numerical value is the canonical one
    The transformation taken to reach the canonical variant is recorded with a code:
        - 0 = identity
        - 1 = h_flip
        - 2 = v_flip
        - 3 = rot90
        - 4 = rot90, h_flip
        - 5 = rot90, v_flip
        - 6 = rot90, rot90
        - 7 = rot90, rot90, rot90
    '''
    h_flip, v_flip, rot90 = maps

    canon_trans = {0:0}
    
    group_len = len(groups[pivot_ind])

    # iterate through each combination of bits in the pivot
    for i in range(1, 1 << group_len):
        variants = []
        pos = i

        # find all variant of this position
        for j in range(2):
            variants.append(pos)
            variants.append(apply_map(pos, h_flip))
            variants.append(apply_map(pos, v_flip))

            pos = apply_map(pos, rot90)

        variants.append(pos)
        variants.append(apply_map(pos, rot90))

        # the index of the variant with the smallest value denotes what transformations need to be applied to get to it
        canon_trans[i] = variants.index(min(variants))
    return canon_trans

def canonise_pos(pos, pivot_size, canon_trans, h_flip, v_flip, rot90):
    '''
    Takes a position and transforms it so the pivot matches the canonical representation
    '''

    # extract the pivot
    pivot_mask = (1 << pivot_size) - 1
    pivot = pos & pivot_mask

    # find the relevant tranformation code for this pivot
    code = canon_trans[pivot]

    # apply the relevant transformation according to the code to return to the canonical representation
    if code < 3:
        if code == 1:
            return apply_map(pos, h_flip)
        elif code == 2:
            return apply_map(pos, v_flip)
        return pos
    else:
        pos = apply_map(pos, rot90)
    
        if code < 6:
            if code == 4:
                return apply_map(pos, h_flip)
            elif code == 5:
                return apply_map(pos, v_flip)
            return pos
        else:
            pos = apply_map(pos, rot90)

            if code == 7:
                return apply_map(pos, rot90)
            return pos

def generate_box_checks(board: Board, groups, group_map):
    '''
    Constructs 2 masks for each line to check the other surrounding lines for each box it is attached to
    Lines on edges are only attached to 1 box so 1 mask will be None
    check_A checks top and left boxes, check_B checks bottom and right boxes
    '''
    n_lines = board.N_LINES
    size = board.SIZE

    check_A = [None] * n_lines
    check_B = [None] * n_lines

    # find bit shifts for each group
    shifts = []
    start = 0
    for i in range(len(groups)):
        shifts.append(start)
        start += len(groups[i])

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
                        group_ind = group_map[lines[j]]
                        mask |= (1 << shifts[group_ind] + groups[group_ind].index(lines[j]))
                
                # a is if box is top or left, b is if box is bottom or right
                if i % 2 == 0:
                    check_B[current_line] = mask
                else:
                    check_A[current_line] = mask
                    
    return check_A, check_B

def distribution_iterator(remaining, capacities, distribution=[]):
    '''
    Recursively iterates through each distribution of missing lines between each group
    '''
    
    # if there are no more groups to fill and there are no more remaining lines, return
    if not capacities:
        if remaining == 0:
            yield tuple(distribution)
        return

    current_capacity = capacities[0]
    remaining_capacities = capacities[1:]

    remaining_capacity = sum(remaining_capacities)
    
    # iterate through each possible assigned number of lines
    for i in range(min(current_capacity, remaining) + 1):
        
        # if there are too many lines to distribute to the rest, this isn't valid
        if remaining_capacity >= remaining - i:
            # perform the same logic to the rest of the groups with the lines already assigned to this one
            yield from distribution_iterator(remaining - i, remaining_capacities, distribution + [i])

# From https://rosettacode.org/wiki/Gosper's_hack#Python
def gospers_hack(ones, bits):
    '''
    Iterates through every combination of n bits set to 1 for a given bit length
    '''
    if ones == 0:
        yield 0
        return
    
    c = (1 << ones) - 1
    limit = (1 << bits)
    
    while c < limit:
        yield c
        x = c
        c = x & -x
        r = x + c
        c = (((r ^ x) >> 2) // c) | r

def group_combinations_iterator(distribution, group_bits, canon_trans):
    '''
    Iterates through every combination of bits where the number of 1s per group adheres to the distribution
    '''

    # only canonical positions are considered
    pivot_combinations = []
    for combination in gospers_hack(distribution[0], group_bits[0]):
        if canon_trans[combination] == 0:
            pivot_combinations.append(combination)

    # each list holds all the possible values for a group that adhere to the distribution of 1s
    group_combinations = [pivot_combinations]
    for ones, bits in zip(distribution[1:], group_bits[1:]):
        group_combinations.append(list(gospers_hack(ones, bits)))

    # iterate through the cartesian product for all possible values each group could take
    for combination in itertools.product(*group_combinations):
        yield combination

def next_pos_iterator(pos, groups, group_sizes):
    '''
    Iterates through every position yielded from removing a single 1
    '''
    temp = pos

    # while there are still 1s that can be removed
    while temp > 0:
        pivot_change = False

        # find the location of the rightmost unprocessed 1
        missing = temp & -temp

        # find what line index that 1 corresponds to
        ind = missing.bit_length() - 1
        for i in range(len(group_sizes)):
            if ind >= group_sizes[i]:
                ind -= group_sizes[i]
            else:
                line = groups[i][ind]

                # if the line added is in the pivot group, label the pivot as being changed
                if i == 0:
                    pivot_change = True
                break

        # remove that 1 from the position, marking it as no longer missing
        yield pos ^ missing, line, pivot_change
        temp &= (temp - 1)
