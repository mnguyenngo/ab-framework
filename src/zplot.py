import scipy.stats as scs
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def zplot(cdf=0.95, two_tailed=True):
    """Plots a z distribution with common annotations

    Example:
        zplot(0.95)

        zplot(0.95, align='left')

    Parameters:
        cdf (float): The area under the standard normal distribution curve.
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
        left = norm.ppf(0.5 - cdf / 2)
        right = norm.ppf(0.5 + cdf / 2)
        ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
        ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')

        ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                        where=(x > left) & (x < right))
        plt.xlabel('z')
        plt.ylabel('PDF')
        plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), fontsize=12,
                 rotation=90, va="bottom", ha="right")

    else:
        right = norm.ppf(cdf)
        ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
        ax.fill_between(x, 0, y, color='grey', alpha='0.25', where=x < right)

    plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right), fontsize=12,
             rotation=90, va="bottom", ha="left")
    plt.text(0, 0.1, "shaded area = {0:.3f}".format(cdf), fontsize=12,
             ha='center')
    plt.xlabel('z')
    plt.ylabel('PDF')
    plt.show()
