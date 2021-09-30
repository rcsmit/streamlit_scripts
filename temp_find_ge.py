my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

import bisect

def find_ge(a, low, high):
    i = bisect.bisect_left(a, low)
    g = bisect.bisect_right(a, high)
    if i != len(a) and g != len(a):
        return a[i:g]
    else:
        # print ("value error")
        # return a[i:g]

        raise ValueError

print (find_ge (my_list, 10,10))
