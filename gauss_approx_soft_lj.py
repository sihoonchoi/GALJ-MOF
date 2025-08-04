import numpy as np
from sklearn.linear_model import LinearRegression

with open('./lj_params.def', 'r') as f:
    lines = f.readlines()

lamda = 0.3
alpha = 0.5
r = np.linspace(0.00, 8, 1000)
gauss_sigmas = np.linspace(0.5, 2, 21)
gauss_gammas = 1 / 2 / gauss_sigmas
X = np.hstack([np.exp(-gauss_gammas * r**2) for s in gauss_sigmas])
model = LinearRegression(fit_intercept = False)

for line in lines:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = sigma / (4 - 2 * alpha * (1 - lamda)**2)**(1/6.)
    soft_lj = 4 * epsilon * ((alpha * (1 - lamda)**2 + (r / sigma_eff)**6)**(-2) - (alpha * (1 - lamda)**2 + (r / sigma_eff)**6)**(-1))

    model.fit(X, soft_lj)
    amplitudes = model.coef_

    with open(f'./gauss_params/soft_lj/{elem}.g', 'w') as f:
        for a, g in zip(amplitudes, gauss_gammas):
            f.write(f'{a}\t{g}\n')
