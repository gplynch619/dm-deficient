'''
Given a size and dimension, this routine optimizes the grid topology of a redistributor

Let S be the size and D the dimension. The basic problem is to find a D-tuple of
integers such that a_1 * ... * a_D = S, where we may assume WLOG that
 a_1 <= a_2 ... <= a_D (note that equality obtains where a_i = S^(1/D).
Also note that we only need to find D-1 numbers since a_D = S/(a_1 * ... * a_(D-1))

For small S (~~<10^5, so most applications) this can be solved roughly as follows:
    1. Get a set of factors F of S.
    2. Find all (D-1)-tuples {a_i} in F such that S/(a_1 * ... * a_(D-1)) is in F
    3. If there are multiple such tuples, return the one with smallest a_D - a_1

To be concrete, for D = 3, we want to find all triplets a, b, c such that both
a, b are in F and S/(a*b) is in F. If there are multiple, we would like the one
with smallest c - a. 

As written, I think this is O(S^D) :O 

'''

import itertools

def topology_optimizer(size, dim):
    
    factors = set()
    for i in range(1, int(size ** 0.5)+1):
        divisor, remainder = divmod(size, i)
        if remainder == 0:
            factors |= {i, divisor}
    
    #the set factors is an unordered set of the factors of size

    possible_tuples = itertools.product(list(factors), repeat=dim-1) #generates an iterator with all dim-1 tuples of factors

    duplicate_list = []
    for tup in possible_tuples:
#        prod = reduce(lambda x,y : x*y, tup)
        prod = 1
        for num in tup:
            prod *= num
        ff = size/float(prod)
        if ff in factors:
            duplicate_list.append(tup+(int(ff),))

    # now we remove duplicate tuples
    final_list = list(set(tuple(sorted(l)) for l in duplicate_list))

    x = size - 1
    fi = 0
    # now we find solution with most even spread across dimensions
    for i, tup in enumerate(final_list):
        diff = tup[-1] - tup[0]
        if diff < x:
            x = diff
            fi = i
    return list(final_list[fi])
    
if __name__ == "__main__":
    while True:
        num = input("What number should I factor?\n")
        print(topology_optimizer(int(num), 3))
