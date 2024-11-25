# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def power_law_component(a=2, b=3):
    t = np.linspace(1, 3, 5000)
    fig, ax = plt.subplots(2, 1, figsize=(7, 4))
    for m in np.linspace(0.01, 1. - 0.0001, 30):
        logp = a + b * t ** m
        p = np.exp(logp)
        if m > 0.8:
            ax[0].plot(t, p, label=f"m={m:.2f}")
            ax[1].plot(t, logp, label=f"m={m:.2f}")
        else:
            ax[0].plot(t, p)
            ax[1].plot(t, logp)
    for adx in range(2):
        ax[adx].legend(loc="upper left")
        ax[adx].set_xlabel("t")
        ax[0].set_title(
            r'$log(price)='
            r'{} {} {}t^m$'.format(a, "+" if b >= 0 else "", b),
            fontsize=16, color='k')

    ax[0].set_ylabel(r'price')
    ax[1].set_ylabel(r'$\log(price)$')

    plt.show()


def log_periodic(cs=[0.5, 1, 1.5, 2], omegas=[8, 16, 24, 32],
                 phis=[100, 200, 300, 400], ms=[0.25, 0.5, 0.75, 0.90]):
    fig, ax = plt.subplots(4, 1, figsize=(7, 4))
    tc = 500
    t = np.linspace(1, tc - 0.1, tc)  # [::-1]
    for c in cs:
        omega = omegas[0]
        phi = phis[0]
        m = ms[0]
        ax[0].plot(t, c * (tc - t) ** (m) * np.cos(omega * np.log(tc - t) - phi),
                   label=f"c={c:.2f}")
        ax[0].legend(loc="lower left")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("price")
    ax[0].set_title(
        r'$\log(price)='
        r'c({tc}-t)^{m:.2f} \cos({omega} \log({tc}-t)-{phi})$, m={m:.2f}'.format(
            tc=tc, omega=omega, phi=phi, m=m),
        # fontsize=14,
        color='k')

    for omega in omegas:
        c = cs[0]
        phi = phis[0]
        m = ms[0]
        ax[1].plot(t, c * (tc - t) ** (m) * np.cos(omega * np.log(tc - t) - phi),
                   label=f"$\omega$={omega:.2f}")
        ax[1].legend(loc="lower left")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("price")
    ax[1].set_title(
        r'$\log(price)='
        r'{c}({tc}-t)^m \cos(\omega \log({tc}-t)-{phi})$, m={m:.2f}'.format(
            c=c, tc=tc, phi=phi, m=m),
        # fontsize=14,
        color='k')

    for phi in phis:
        c = cs[0]
        omega = omegas[0]
        m = ms[0]
        ax[2].plot(t, c * (tc - t) ** (m) * np.cos(omega * np.log(tc - t) - phi),
                   label=f"$\phi$={phi:.2f}")
        ax[2].legend(loc="lower left")
        ax[2].set_xlabel("t")
        ax[2].set_ylabel("price")
    ax[2].set_title(
        r'$\log(price)='
        r'{c}({tc}-t)^m \cos({omega} \log({tc}-t)-\phi)$, m={m:.2f}'.format(
            c=c, omega=omega, tc=tc, m=m),
        # fontsize=14,
        color='k')

    for m in ms:
        c = cs[0]
        omega = omegas[0]
        phi = phis[0]
        ax[3].plot(t, c * (tc - t) ** (m) * np.cos(omega * np.log(tc - t) - phi),
                   label=f"$m$={m:.2f}")
        ax[3].legend(loc="lower left")
        ax[3].set_xlabel("t")
        ax[3].set_ylabel("price")
    ax[3].set_title(
        r'$\log(price)='
        r'{c}({tc}-t)^m \cos({omega} \log({tc}-t)-{phi})$'.format(
            c=c, omega=omega, tc=tc, phi=phi),
        # fontsize=14,
        color='k')

    # plt.tight_layout()
    # wspace 和 hspace 指定子圖之間保留的空間。它們分別是軸的寬度和高度的分數。
    plt.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.5)
    plt.show()


if __name__ == '__main__':
    # power_law_component(2, -3)
    log_periodic()
    # log_periodic(c=5)
