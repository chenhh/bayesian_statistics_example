# -*- coding: UTF-8 -*-
import sympy as sb

def binomial_dist_mle():
    theta = sb.symbols('theta')
    x, p, k, n = sb.symbols('x p k n')
    f = p**k * (1-p)**(n-k)
    print(f)



if __name__ == '__main__':
    binomial_dist_mle()
