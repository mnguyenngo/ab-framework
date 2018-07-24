import scipy.stats as scs
import pandas as pd
# import numpy as np


def generate_data(N_A, N_B, bcr, d_hat, days=None, control_label='A',
                  test_label='B'):
    """Returns a pandas dataframe with fake CTR data
    """

    # A = scs.bernoulli(bcr).rvs(N_A)
    # B = scs.bernoulli(bcr+d_hat).rvs(N_B)
    #
    data = []
    # A_data['conversion'] = A
    # B_data['conversion'] = B

    N = N_A + N_B

    for idx in range(N):
        row = {}
        if days is not None:
            if type(days) == int:
                row['ts'] = idx // (N // days)
            else:
                raise ValueError("Provide int for the days parameter.")
        row['group'] = scs.bernoulli(0.5).rvs()
        if row['group'] == 1:
            row['converted'] = scs.bernoulli(bcr+d_hat).rvs()
        else:
            row['converted'] = scs.bernoulli(bcr).rvs()

        data.append(row)

    df = pd.DataFrame(data)
    df['group'] = df['group'].apply(
        lambda x: control_label if x == 0 else test_label)

    return df
