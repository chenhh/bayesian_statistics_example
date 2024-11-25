# -*- coding: UTF-8 -*-
import math


def stirling_approximation(n: int) -> int:
    return math.sqrt(2 * n * math.pi) * pow(n, n) * math.exp(-n)


if __name__ == '__main__':
    for n in range(1, 50):
        fact = math.factorial(n)
        approx = stirling_approximation(n)
        diff = fact - approx
        print(f"n={n}, n!={fact}, diff:{diff:.2f}")
