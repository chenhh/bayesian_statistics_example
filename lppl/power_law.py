# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def power_law_component():
    t = np.linspace(1, 3, 5000)
    fig = plt.figure(figsize=(7, 4))
    for m in np.linspace(0.01, 1. - 0.0001, 30):
        logp = 2 + 3 * t ** m
        p = np.exp(logp)
        if m > 0.8:
            plt.plot(t, p, label=f"m={m:.2f}")
        else:
            plt.plot(t, p)
    plt.legend(loc="upper left")
    plt.xlabel("t")
    plt.ylabel(r'price')
    plt.gca().set_title(r'$log(price)='
                        r'A+Bt^m$', fontsize=16, color='k')
    fig.show()


if __name__ == '__main__':
    power_law_component()
