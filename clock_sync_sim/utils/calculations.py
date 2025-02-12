import numpy as np
from scipy.special import binom
from clock_sync_sim.config.settings import SIGMA
from math import factorial as f

# Efficiency and BS parameters
eta_H_A = 1
eta_V_A = 1
eta_H_B = 1
eta_V_B = 1
T = R = 0.5

# Functional definition of spectral envelopes
def get_shape(shape, sigma, w, wb):
    shapes = {"gaussian": 1 / ((2 * np.pi) ** 0.25 * np.sqrt(sigma)) * np.exp(-(w - wb) ** 2 / (4 * sigma ** 2)),
              "sinc": np.sqrt(1 / sigma) * np.sinc(1 / sigma * (w - wb)),
              "lorentz": np.sqrt(2 * sigma / np.pi) * (sigma / ((w - wb) ** 2 + (sigma) ** 2)),
              "sech": np.sqrt(0.5 / sigma) * 1 / np.cosh((w - wb) / sigma)
              }
    return shapes[shape]


# Retreive numerical representation of spectral envelope
def get_envelopes(shapeA, shapeB, sigmaA, sigmaB, wbA, wbB, w, tau):
    # Spec envelope A
    phi = get_shape(shapeA, sigmaA, w, wbA)
    # Spec envelope B with time delay tau
    varphi = get_shape(shapeB, sigmaB, w, wbB) * np.exp(1j * w * tau)

    return phi, varphi


# Calculate spectral overlap
def spec_inner_prod(phi, varphi, w):
    integral = np.trapz(phi * np.conjugate(varphi), x=w) * np.trapz(np.conjugate(phi) * varphi, x=w)

    return np.real(integral)


# Define detector efficiency function
def detector_sensitivity(eta_H, eta_V, pol_A, pol_B, m, n):
    return 1 - (1 - (eta_H * pol_A[0] ** 2 + eta_V * pol_A[1] ** 2)) ** m * (
                1 - (eta_H * pol_B[0] ** 2 + eta_V * pol_B[1] ** 2)) ** n


# Calculate probability of detecting all photons at outport A
def ProbA(pol_A, pol_B, m, n, phi, var_phi, w):
    pol_mismatch = np.dot(pol_A, pol_B)
    spectral_overlap = spec_inner_prod(phi, var_phi, w)

    inner_product = T ** m * R ** n * sum(
        [binom(m, q) * binom(n, q) * pol_mismatch ** (2 * q) * spectral_overlap ** q for q in range(min(m, n) + 1)])
    return inner_product


# Calculate probability of detecting all photons at outport B
def ProbB(pol_A, pol_B, m, n, phi, var_phi, w):
    pol_mismatch = np.dot(pol_A, pol_B)
    spectral_overlap = spec_inner_prod(phi, var_phi, w)

    inner_product = T ** n * R ** m * sum(
        [binom(m, q) * binom(n, q) * pol_mismatch ** (2 * q) * spectral_overlap ** q for q in range(min(m, n) + 1)])
    return inner_product


# Calculate the coincidence probability
def P_Co(m, n, polA, polB, phiA, phiB, w):
    if m == 0 and n == 0:  # No photons -> dark counts
        return 0
    else:
        # Determine efficiency of detector A
        det_func_A = detector_sensitivity(eta_H_A, eta_V_A, polA, polB, m, n)
        # Determine efficiency of detector B
        det_func_B = detector_sensitivity(eta_H_B, eta_V_B, polA, polB, m, n)
        p_co = det_func_A * det_func_B - (
                    det_func_A * ProbA(polA, polB, m, n, phiA, phiB, w) + det_func_B * ProbB(polA, polB, m, n, phiA,
                                                                                             phiB, w))
    return p_co


def get_coin_prob(m, n, mismatch, td):
    return 1 - 1 / (2 ** (m + n - 1)) * sum([
        binom(m, j)* binom(n, j)
        * (mismatch ** (2 * j))
        * np.exp(-0.5 * j * SIGMA ** 2 * (td ** 2))
        for j in range(min(m, n) + 1)
    ])


def calc_spectral_inner_prod(phi_A, phi_B, omega):
    return 1


def p_coherent(muA, muB, polA, polB, phi, varphi, w):
    p_coh = 0

    for m in range(25):
        for n in range(25):
            p_co_mn = P_Co(m, n, polA, polB, phi, varphi, w)
            p_coh += np.exp(-muA-muB)*muA**m * muB**n /f(m)/f(n) * p_co_mn

    return p_coh

# For mixed sourcing, perfect detectors assumed
def calculate_probability(source_A, source_B, photon_A, photon_B, pol_A, pol_B, phi_A, phi_B, omega):
    if source_A == source_B == "coherent":
        mu_A = photon_A
        mu_B = photon_B
        return p_coherent(mu_A, mu_B, pol_A, pol_B, phi_A, phi_B, w)
    elif source_A == "coherent" and source_B == 'spdc':
        mu_A = photon_A
        spectral_inner_prod = spec_inner_prod(phi_A, phi_B, omega)
        pol_mismatch = np.dot(pol_A, pol_B)
        return 1 - np.exp(-mu_A)*np.sum([(0.5*mu_A)**f(n)*(1+n*pol_mismatch**2 * spectral_inner_prod) for n in range(10)])
    elif source_A == "spdc" and source_B == "coherent":
        mu_B = photon_B
        spectral_inner_prod = spec_inner_prod(phi_A, phi_B, omega)
        pol_mismatch = np.dot(pol_A, pol_B)
        return 1 - np.exp(-mu_B)*np.sum([(0.5*mu_B)**f(m)*(1+m*pol_mismatch**2 * spectral_inner_prod) for m in range(10)])
    elif source_A == source_B == "spdc":
        return P_Co(photon_A, photon_B, pol_A, pol_B, phi_A, phi_B, omega)
    else:
        print("Incorrect sources specified.")
        return 0