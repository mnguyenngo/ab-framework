import scipy.stats as scs
import pandas as pd
# import numpy as np


def generate_data(N_A, N_B, bcr, d_hat, days=None, control_label='A',
                  test_label='B'):
    """Returns a pandas dataframe with fake CTR data

    Example:

    Parameters:
        N_A (int): sample size for control group
        N_B (int): sample size for test group
            Note: final sample size may not match N_A provided because the
            group at each row is chosen at random (50/50).
        bcr (float): base conversion rate; conversion rate of control
        d_hat (float): difference in conversion rate between the control and
            test groups, sometimes referred to as **minimal detectable effect**
            when calculating minimum sample size or **lift** when discussing
            positive improvement desired from launching a change.
        days (int): optional; if provided, a column for 'ts' will be included
            to divide the data in chunks of time
            Note: overflow data will be included in an extra day
        control_label (str)
        test_label (str)

    Returns:
        df (df)
    """

    # initiate empty container
    data = []

    # total amount of rows in the data
    N = N_A + N_B

    for idx in range(N):
        # initite empty row
        row = {}
        # for 'ts' column
        if days is not None:
            if type(days) == int:
                row['ts'] = idx // (N // days)
            else:
                raise ValueError("Provide an integer for the days parameter.")
        # assign group based on 50/50 probability
        row['group'] = scs.bernoulli(0.5).rvs()
        if row['group'] == 1:
            # assign conversion based on provided parameters
            row['converted'] = scs.bernoulli(bcr+d_hat).rvs()
        else:
            row['converted'] = scs.bernoulli(bcr).rvs()
        # collect row into data container
        data.append(row)

    # convert data into pandas dataframe
    df = pd.DataFrame(data)

    # transform group labels of 0s and 1s to user-defined group labels
    df['group'] = df['group'].apply(
        lambda x: control_label if x == 0 else test_label)

    return df
