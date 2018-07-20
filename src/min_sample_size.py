import scipy.stats as scs


def min_sample_size(prob_control, effect_size, power=0.8, sig_level=0.05):
    """Returns the minimum sample size to set up a split test

    Arguments:
        prob_control (float): probability of success for control, sometimes
        referred to as baseline conversion rate

        effect_size (float): minimum change in measurement between control
        group and test group if alternative hypothesis is true

        power (float): probability of rejecting the null hypothesis when the
        null hypothesis is false, typically 0.8

        sig_level (float): significance level often denoted as alpha,
        typically 0.05

    Returns:
        min_N: minimum sample size (float)

    References:
        Stanford lecture on sample sizes
        http://statweb.stanford.edu/~susan/courses/s141/hopower.pdf
    """
    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)

    # find Z_beta from desired power
    Z_beta = standard_norm.ppf(power)

    # find Z_alpha
    Z_alpha = standard_norm.ppf(1-sig_level/2)

    pooled_prob = prob_control + effect_size / 2

    min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2
             / effect_size**2)

    return min_N
