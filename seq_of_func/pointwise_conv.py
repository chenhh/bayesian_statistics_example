# -*- coding: UTF-8 -*-

def disc_conti():
    import numpy as np
    import matplotlib.pyplot as plt
    xs = np.linspace(-1.1, 1.1, 1000)

    for n in range(1, 50):
        exp = xs**(2*n)
        funcs = exp/(1+exp)
        plt.plot(xs, funcs, label=f"n={n}")

    # plt.legend()
    plt.show()


if __name__ == '__main__':
    disc_conti()
