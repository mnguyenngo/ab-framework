import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import scipy.stats as scs

from .stats import pooled_SE

plt.style.use('ggplot')


def plot_norm_dist(ax, mu, sig):
    """Adds a normal distribution to the axes provided

    Example:
        plot_norm_dist(ax, 0, 1)  # plots a standard normal distribution

    Parameters:
        ax (matplotlib axes)
        mu (float): mean of the normal distribution
        sig (float): standard deviation of the normal distribution

    Returns:
        None: the function adds a plot to the axes object provided
    """
    x = np.linspace(mu - 6 * sig, mu + 6 * sig, 1000)
    y = scs.norm(mu, sig).pdf(x)
    ax.plot(x, y)


def plot_CI(ax, mu, s, sig_level=0.05, color='grey'):
    """Calculates the two-tailed confidence interval and adds the plot to
    an axes object.

    Example:
        plot_CI(ax, mu=0, s=stderr, sig_level=0.05)

    Parameters:
        ax (matplotlib axes)
        mu (float): mean
        s (float): standard deviation

    Returns:
        None: the function adds a plot to the axes object provided
    """
    z = scs.norm().ppf(1 - sig_level/2)
    left = mu - z * s
    right = mu + z * s
    ax.axvline(left, c=color, linestyle='--', alpha=0.5)
    ax.axvline(right, c=color, linestyle='--', alpha=0.5)


def plot_control(ax, stderr):
    """Plots the null hypothesis distribution where if there is no real change,
    the distribution of the differences between the test and the control groups
    will be normally distributed.

    The confidence band is also plotted.

    Example:
        plot_control(ax, stderr)

    Parameters:
        ax (matplotlib axes)
        stderr (float): the pooled standard error of the control and test group

    Returns:
        None: the function adds a plot to the axes object provided

    """
    plot_norm_dist(ax, 0, stderr)
    plot_CI(ax, mu=0, s=stderr, sig_level=0.05)


def plot_test(ax, stderr, d_hat):
    """Plots the alternative hypothesis distribution where if there is a real
    change, the distribution of the differences between the test and the
    control groups will be normally distributed and centered around d_hat

    The confidence band is also plotted.

    Example:
        plot_test(ax, stderr, d_hat=0.025)

    Parameters:
        ax (matplotlib axes)
        stderr (float): the pooled standard error of the control and test group

    Returns:
        None: the function adds a plot to the axes object provided
    """
    plot_norm_dist(ax, d_hat, stderr)
    plot_CI(ax, d_hat, stderr, sig_level=0.05)


def abplot(n, bcr, d_hat):
    """Example plot of AB test

    Example:
        abplot(n=4000, bcr=0.11, d_hat=0.03)

    Parameters:
        n (int): total sample size for both control and test groups (N_A + N_B)
        bcr (float): base conversion rate; conversion rate of control
        d_hat: difference in conversion rate between the control and test
            groups, sometimes referred to as **minimal detectable effect** when
            calculating minimum sample size or **lift** when discussing
            positive improvement desired from launching a change.

    Returns:
        None: the function plots an AB test as two distributions for
        visualization purposes
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    N_A = n / 2
    N_B = n / 2
    X_A = bcr * N_A
    X_B = (bcr + d_hat) * N_B
    stderr = pooled_SE(N_A, N_B, X_A, X_B)
    plot_control(ax, stderr)
    plot_test(ax, stderr, d_hat)
    ax.set_xlim(-3*d_hat, 3*d_hat)
    plt.show()


def zplot(area=0.95, two_tailed=True, align_right=False):
    """Plots a z distribution with common annotations

    Example:
        zplot(area=0.95)

        zplot(area=0.95, align='left')

    Parameters:
        area (float): The area under the standard normal distribution curve.
        align (str): The area under the curve can be aligned to the center
            (default) or to the left.

    Returns:
        None: A plot of the normal distribution with annotations showing the
        area under the curve and the boundaries of the area.
    """

    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots()
    norm = scs.norm()

    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x)

    ax.plot(x, y)

    if two_tailed:
        left = norm.ppf(0.5 - area / 2)
        right = norm.ppf(0.5 + area / 2)
        ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
        ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')

        ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                        where=(x > left) & (x < right))
        plt.xlabel('z')
        plt.ylabel('PDF')
        plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), fontsize=12,
                 rotation=90, va="bottom", ha="right")
        plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                 fontsize=12, rotation=90, va="bottom", ha="left")

    else:
        if align_right:
            left = norm.ppf(1-area)
            ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                            where=x > left)
            plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left),
                     fontsize=12, rotation=90, va="bottom", ha="right")
        else:
            right = norm.ppf(area)
            ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                            where=x < right)
            plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                     fontsize=12, rotation=90, va="bottom", ha="left")

    plt.text(0, 0.1, "shaded area = {0:.3f}".format(area), fontsize=12,
             ha='center')
    plt.xlabel('z')
    plt.ylabel('PDF')
    plt.show()
