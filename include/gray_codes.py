#!/bin/python
import sys

def gray_code(n):
    def gray_code_recurse (g,n):
        k=len(g)
        if n<=0:
            return

        else:
            for i in range (k-1,-1,-1):
                char='1'+g[i]
                g.append(char)
            for i in range (k-1,-1,-1):
                g[i]='0'+g[i]

            gray_code_recurse (g,n-1)

    g=['0','1']
    gray_code_recurse(g,n-1)
    return g

def main():
    n= int(sys.argv[1])
    g = gray_code (n)

    print(g)
    add_or_subtract = list()
    add_or_subtract.append(0)
    diff_idxs = list()
    diff_idxs.append(0)
    for i in range(len(g) - 1):
      s1 = g[i]
      s2 = g[i+1]
      # print("s1: {}, s2: {}".format(s1, s2))
      diff = [i for i in xrange(len(s1)) if s1[i] != s2[i]]
      assert(len(diff) == 1)
      diff_idx = diff[0]
      diff_idxs.append(n - 1 - diff_idx)

      if (s1[diff_idx] == '0' and s2[diff_idx] == '1'):
        add_or_subtract.append(1)
      else:
        add_or_subtract.append(0)
      # print(diff)

    print("gray code idx diffs:")
    print(diff_idxs)

    print("\nadd or subtract:")
    print(add_or_subtract)
    '''
    if n>=1:
        for i in range (len(g)):
            print g[i],
    '''

main()
