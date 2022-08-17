# -*- coding: UTF-8 -*-
import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt
from typing import Iterable
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.ticker import FormatStrFormatter


def variance(values: Iterable):
    xs = np.asarray(values)

    e_x = xs.mean()
    e_x2 = (xs * xs).mean()
    var_x1 = e_x2 - e_x * e_x
    var_x2 = ((xs - e_x) * (xs - e_x)).mean()
    var_np = xs.var(ddof=0)  # default bias estimator
    np.testing.assert_almost_equal(var_x1, var_np, decimal=10)
    np.testing.assert_almost_equal(var_x2, var_np, decimal=10)
    print(f"{var_x1}, {var_x2}, {var_np}")


def probability_mass_function(values: Iterable):
    ys = np.asarray(values)
    fig, ax = plt.subplots()
    n_bin = ys.size // 10
    n, bins, patches = ax.hist(ys, bins=n_bin, density=True, cumulative=False, edgecolor='black', label='pmf')
    ax.set_ylabel('Frequency')
    ax.set_title('Probability')
    ax.legend(fontsize=24)
    fig.tight_layout()
    plt.show()


def empirical_distribution(values: Iterable):
    ys = np.asarray(values)
    fig, ax = plt.subplots()
    n_bin = ys.size // 10
    n, bins, patches = ax.hist(ys, bins=n_bin, density=True, cumulative=True, label='empirical distribution')
    ax.set_ylabel('Frequency')
    ax.set_title('Probability')
    ax.legend()
    fig.tight_layout()
    plt.show()
    print(matplotlib.get_backend())


def kth_moment(values: Iterable):
    ys = np.asarray(values)
    mu = ys.mean()
    for k in range(1, 4 + 1):
        e_xs = (ys ** k).mean()
        e_center_xs = ((ys - mu) ** k).mean()
        print(f"{k}th moment: {e_xs:.4f}")
        print(f"{k}th center moment: {e_center_xs:.4f}")


def sample_distribution(n_sample=5000, n_point=10000):
    mus = np.random.geometric(0.2, (n_sample, n_point)).mean(axis=1)  # size: n_sample
    vars = np.random.geometric(0.2, (n_sample, n_point)).var(axis=1)
    fig, ax = plt.subplots(2, 1)
    n_bin = n_sample // 10
    ax[0].hist(mus, bins=n_bin, density=True, edgecolor='black', label='sample distribution of mean')
    ax[1].hist(vars, bins=n_bin, density=True, edgecolor='black', color='green', label='sample distribution of var')
    ax[0].set_ylabel('Frequency')
    ax[1].set_ylabel('Frequency')
    fig.suptitle('Sample distribution', fontsize=24)
    ax[0].legend(fontsize=20)
    ax[1].legend(fontsize=20)
    fig.tight_layout()
    plt.show()


def binomial_distribution(n_point=10000, n=100, p=0.2):
    values = np.random.binomial(n, p, n_point)
    q = 1 - p
    mu = values.mean()
    var = values.var()
    skew = spstats.skew(values)
    kurt = spstats.kurtosis(values)

    # skewness of scipy, Fisher-Pearson coefficient
    m2 = ((values - mu) ** 2).mean()  # variance
    m3 = ((values - mu) ** 3).mean()
    g1 = m3 / (m2 ** 1.5)
    g1f = (q - p) / np.sqrt(n * p * q)  # 偏度公式與樣本計算值偏差大

    # kurtosis of scipy
    m4 = ((values - mu) ** 4).mean()
    g2 = m4 / m2 / m2 - 3
    g2f = (1 - 6 * p * q) / (n * p * q)  # 峰度公式與樣本計算值偏差大
    print(f"{mu}, var:{var}, m2:{m2} "
          f"skew: {skew}, {g1}, {g1f}, "
          f"kurt:{kurt}, {g2}, {g2f}")

    np.testing.assert_approx_equal(mu, n * p, significant=2)
    np.testing.assert_approx_equal(var, n * p * (1 - p), significant=2)
    np.testing.assert_approx_equal(var, m2, significant=7)
    np.testing.assert_approx_equal(skew, g1, significant=7)
    np.testing.assert_approx_equal(kurt, g2, significant=7)

    fig, ax = plt.subplots()
    ax.hist(values, bins=50, density=True, label='binomial distribution')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Binomial (N=100, p=0.2) $\mu = {mu:.2f}, \sigma^2={var:.2f}$')
    ax.legend(fontsize=20)
    fig.tight_layout()
    plt.show()


def independent_binomial_dist(n_point=10000, n1=100, n2=500, p=0.9):
    x1s = np.random.binomial(n1, p, n_point)
    x2s = np.random.binomial(n2, p, n_point)
    x12s = np.random.binomial(n1 + n2, p, n_point)

    for k in range(1, 6):
        mk1 = spstats.moment(x1s, k)
        mk2 = spstats.moment(x2s, k)
        mk12 = spstats.moment(x12s, k)
        print(f"{k}th center moment: {mk1}, {mk2}, {mk1 + mk2}, {mk12}")


def central_limit_theorem(n_sample=1000, n_point=10000):
    for _ in range(20):
        a, b = np.random.randint(1, 100), np.random.randint(1, 100)
        values = np.random.beta(a, b, (n_sample, n_point))
        mu = a / (a + b)
        std = np.sqrt((a * b) / ((a + b) * (a + b) * (a + b + 1)))

        mu_hat = values.mean(axis=1)

        zs = np.sqrt(n_point) * (mu_hat - mu) / std
        z_mean, z_std = zs.mean(), zs.std()
        z_skew, z_kurt = spstats.skew(zs), spstats.kurtosis(zs)
        print(f"beta({a}, {b}): mu:{z_mean:.2f}, std:{z_std:.2f}, "
              f"skew:{z_skew:.2f}, kurt:{z_kurt:.2f}")


def white_noise(n_point=10000):
    from statsmodels.tsa.stattools import (acovf, acf)
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    values = np.random.rand(n_point)
    gammas = acovf(values)
    rhos = acf(values)
    print(f"gamma:{gammas}, rho:{rhos}")
    plot_acf(values)
    plt.show()


if __name__ == '__main__':
    # values = np.random.randn(100000)
    # variance(values)
    # probability_mass_function(values)
    # empirical_distribution(values)
    # kth_moment(values)
    # sample_distribution()
    # binomial_distribution()
    # independent_binomial_dist()
    # central_limit_theorem()
    white_noise()
