import numpy as np
import pylab as plt
import os
import warnings

from scipy.optimize import minimize

warnings.simplefilter('ignore')

# defining functions

def get_result(x0, r):
    n = int(len(x0) / 2)
    result = np.zeros(len(r))
    for i in range(n):
        result += np.exp(-1 / 2 / x0[i]**2 * r**2) * x0[i + n]
    return result

def mse(x0, r, V):
    n = int(len(x0) / 2)
    result = np.zeros(len(r))
    
    for i in range(n):
        result += np.exp(-1 / 2 / x0[i]**2 * r**2) * x0[i + n]
        
    return np.mean(np.square(result-V))

def get_LJ(sigma, epsilon, r):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def plot_result(x, elem, sigma, epsilon):
    fig, axes = plt.subplots(3, 1, figsize = (10, 8), dpi = 150, sharex = True, gridspec_kw = {'height_ratios': [3, 1, 2]})

    r = np.linspace(1e-5, 7, 1000)
    G = get_result(x, r) * R / 1000
    LJ = get_LJ(sigma, epsilon, r) * R / 1000
    
    gauss = np.zeros(LJ.shape)
    for i in range(8):
        s, = axes[0].plot(r, x[i + 8] * np.exp(-1 / 2 / x[i]**2 * r **2) * R / 1000, '--', c = '0.5', lw = 0.7)
    
    g, = axes[0].plot(r, G, 'r-', lw = .8)
    l, = axes[0].plot(r[::10], LJ[::10], 'r.', markersize = '4')
    axes[0].plot([0, 7], [0, 0], c = '0.2', lw = 0.4, alpha = 2)
    
    axes[0].set_ylim([-8, 8])
    axes[0].set_yticks(np.linspace(-8, 8, 5))
    axes[0].set_ylabel('U [kJ/mol]', fontsize = 'x-large')
    axes[0].yaxis.set_label_coords(-0.07, 0.5, transform = axes[0].transAxes)
    
    e, = axes[1].plot(r, abs(G - LJ), 'b-')
    axes[2].plot(r, abs(G- LJ), 'b-')

    axes[2].set_xlabel('r [$\mathrm{\AA}$]', fontsize = 'x-large')
    axes[2].set_ylabel('U [kJ/mol]', fontsize = 'x-large')
    axes[2].yaxis.set_label_coords(-0.07, 0.8125, transform = axes[2].transAxes)
    
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    axes[1].set_xlim([-.1, 7])
    axes[1].set_ylim([1e62, 1e70])
    axes[2].set_ylim([1e-6, 1e10])

    axes[1].set_yticks([1e64, 1e68])
    axes[2].set_yticks(np.logspace(-4, 8, 4))

    axes[1].spines['bottom'].set_visible(False)
    axes[1].xaxis.tick_top()
    axes[1].tick_params(labeltop = False)

    axes[2].spines['top'].set_visible(False)
    axes[2].xaxis.tick_bottom()

    d = 0.005
    kwargs = dict(transform = axes[1].transAxes, color = 'k', clip_on = False)
    axes[1].plot((-d, +d), (0, 0), **kwargs)
    axes[1].plot((1 - d, 1 + d), (0, 0), **kwargs)

    kwargs.update(transform = axes[2].transAxes)
    axes[2].plot((-d, +d), (1, 1), **kwargs)
    axes[2].plot((1 - d, 1 + d), (1, 1), **kwargs)
    
    axes[0].text(0.01, 0.94, '(a)', transform = axes[0].transAxes, fontsize = 'x-large')
    axes[1].text(0.01, 0.84, '(b)', transform = axes[1].transAxes, fontsize = 'x-large')
    
    axes[0].legend([s, g, l], ['single Gaussian', 'sum of Gaussians', 'LJ potential'], fontsize = 'x-large')
    axes[1].legend([e], ['absolute error'], fontsize = 'large')
    
    plt.tight_layout()
    plt.savefig('./figures/{}.png'.format(elem))
    plt.show()
    plt.close()
    
def save_gaussian(x0, elem):
    result_filename = "./gaussian_params/{}.g".format(elem)
    
    f = open(result_filename, "w")
    n = int(len(x0) / 2)

    for i in range(n):
        f.write("{}\t{}\n".format(x0[i + n] * R / 10, 1 / 2 / x0[i]**2))
    f.close()
    
    return

def get_optimized(elem, sigma, epsilon, x0, r, plot = False):
    print('optimizing {}\n'.format(elem))

    V = get_LJ(sigma, epsilon, r)
    
    temp1 = list(np.linspace(0.5, 1, 5)) + list(np.linspace(10, 20, 3))
    temp2 = list(np.logspace(10, 6, 4)) + list(-np.logspace(3, 1, 4))
    
    x0 = np.concatenate((temp1, temp2))
    
    res = minimize(mse, x0, (r, V), method='nelder-mead',
              options={'maxiter': 50000, 'disp': True, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive':True})
    
    if plot == True:
        plot_result(res.x, elem, sigma, epsilon)
        
    return res.x

R = 8.31446261815324

if not os.path.isdir('./figures'):
    os.makedirs('./figures')

if not os.path.isdir('./gaussian_params'):
    os.makedirs('./gaussian_params')

with open('./LJ_params.def', 'r') as f:
    lines = f.readlines()

# optimizing O

for line in lines[0:1]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.5, r_min + 6.0, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing N

for line in lines[1:2]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.6, r_min + 6, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(10, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing C

for line in lines[2:3]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)
    
    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.6, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(10, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing F

for line in lines[3:4]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 6.0, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(10, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing B

for line in lines[4:5]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing I

for line in lines[5:6]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.9, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing P

for line in lines[6:7]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing S

for line in lines[7:8]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.4, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)


# optimizing W

for line in lines[8:9]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.75, r_min + 5.3, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Y

for line in lines[9:10]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing K

for line in lines[10:11]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Cl

for line in lines[11:12]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.3, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Br

for line in lines[12:13]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.6, r_min + 5.4, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing H

for line in lines[13:14]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.2, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Zn

for line in lines[14:15]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.5, r_min + 5.4, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(3, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Be

for line in lines[15:16]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.5, r_min + 5.55, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Ca

for line in lines[16:17]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.5, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Cr

for line in lines[17:18]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.6, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Fe

for line in lines[18:19]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.6, r_min + 5.4, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Mn

for line in lines[19:20]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Cu

for line in lines[20:21]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.9, r_min + 5.6, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Co

for line in lines[21:22]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Ga

for line in lines[22:23]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Ti

for line in lines[23:24]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Sc

for line in lines[24:25]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.55, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing V

for line in lines[25:26]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.4, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Ni

for line in lines[26:27]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.65, r_min + 5.8, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Zr

for line in lines[27:28]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.4, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Mg

for line in lines[28:29]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Ne

for line in lines[29:30]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Ag

for line in lines[30:31]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing In

for line in lines[31:32]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Cd

for line in lines[32:33]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.6, r_min + 5.8, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Sb

for line in lines[33:34]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Te

for line in lines[34:35]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.6, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Al

for line in lines[35:36]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.9, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Si

for line in lines[36:37]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing As

for line in lines[37:38]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing La

for line in lines[38:39]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.75, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Ar

for line in lines[39:40]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Au

for line in lines[40:41]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.75, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Rh

for line in lines[41:42]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.6, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Li

for line in lines[42:43]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.5, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Ba

for line in lines[43:44]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Sr

for line in lines[44:45]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.5, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Pd

for line in lines[45:46]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.75, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Mo

for line in lines[46:47]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.75, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Na

for line in lines[47:48]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.45, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Kr

for line in lines[48:49]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.6, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Xe

for line in lines[49:50]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Gd

for line in lines[50:51]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Er

for line in lines[51:52]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.85, r_min + 5.65, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Dy

for line in lines[52:53]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.9, r_min + 5.3, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing U

for line in lines[53:54]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.7, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Tm

for line in lines[54:55]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.9, r_min + 5.65, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Lu

for line in lines[55:56]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.75, r_min + 5.55, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Th

for line in lines[56:57]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.9, r_min + 5.6, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing He

for line in lines[57:58]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.8, r_min + 5.55, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)

# optimizing Pt

for line in lines[58:59]:
    elem = line.split()[0]
    epsilon = float(line.split()[2])
    sigma = float(line.split()[3])

    sigma_eff = (sigma + 3.72) / 2
    epsilon_eff = np.sqrt(epsilon * 158.5)

    r_min = 2**(1/6)*sigma_eff
    r = np.linspace(r_min - 1.7, r_min + 5.55, 1000)

    temp1 = list(np.linspace(1.5, 14, 8))
    temp2 = list(np.logspace(8, 6, 4)) + list(-np.logspace(2, 1, 4))
    x0 = np.concatenate((temp1, temp2))

    x = get_optimized(elem, sigma_eff, epsilon_eff, x0, r, plot = True)
    save_gaussian(x, elem)