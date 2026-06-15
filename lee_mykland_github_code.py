from math import ceil, sqrt
import numpy as np
import pandas as pd

def movmean(v, kb, kf):
    """
    Computes the mean with a window of length kb+kf+1 that includes the element 
    in the current position, kb elements backward, and kf elements forward.
    Nonexisting elements at the edges get substituted with NaN.

    Args:
        v (list(float)): List of values.
        kb (int): Number of elements to include before current position
        kf (int): Number of elements to include after current position

    Returns:
        list(float): List of the same size as v containing the mean values
    """
    m = len(v) * [np.nan]
    for i in range(kb, len(v)-kf):
        m[i] = np.mean(v[i-kb:i+kf+1])
    return m


def LeeMykland(S, sampling, significance_level=0.01):
    """
    "Jumps in Equilibrium Prices and Market Microstructure Noise"
    - by Suzanne S. Lee and Per A. Mykland
    
    "https://galton.uchicago.edu/~mykland/paperlinks/LeeMykland-2535.pdf"
    
    Args:
        S (list(float)): An array containing prices, where each entry 
                         corresponds to the price sampled every 'sampling' minutes.
        sampling (int): Minutes between entries in S
        significance_level (float): Defaults to 1% (0.001)
        
    Returns:
        A pandas dataframe containing a row covering the interval 
        [t_i, t_i+sampling] containing the following values:
        J:   Binary value is jump with direction (sign)
        L:   L statistics
        T:   Test statistics
        sig: Volatility estimate
    """
    tm = 252*24*60  # Trading minutes
    k = ceil(sqrt(tm/sampling))
    r = np.append(np.nan, np.diff(np.log(S)))
    bpv = np.multiply(np.absolute(r[:]), np.absolute(np.append(np.nan, r[:-1])))
    bpv = np.append(np.nan, bpv[0:-1]).reshape(-1, 1)  # Realized bipower variation
    sig = np.sqrt(movmean(bpv, k - 3, 0))  # Volatility estimate
    L = r / sig
    n = np.size(S)  # Length of S
    c = (2 / np.pi)**0.5
    Sn = c * (2 * np.log(n))**0.5
    Cn = (2 * np.log(n))**0.5 / c - np.log(np.pi * np.log(n)) / (2 * c * (2 * np.log(n))**0.5)
    beta_star = -np.log(-np.log(1 - significance_level))  # Jump threshold
    T = (abs(L) - Cn) * Sn
    J = (T > beta_star).astype(float)
    J = J * np.sign(r)  # Add direction
    # First k rows are NaN involved in bipower variation estimation are set to NaN.
    J[0:k] = np.nan
    # Build and return result dataframe
    return pd.DataFrame({'L': L, 'sig': sig, 'T': T, 'J': J})