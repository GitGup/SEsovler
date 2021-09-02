import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time
import functools
%matplotlib widget

# Author/Source: Edward Krueger, https://towardsdatascience.com/a-simple-way-to-time-code-in-python-a9a175eb0172
def timefunc(func):
    """timefunc's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure

class SESolver:

    # GLOBAL CONSTANTS
    H_BAR = 1
    m = 1

    @timefunc
    def __init__(self, start_interval, end_interval, start_time, end_time, psi_size, psi_equ, potential_equ, dt, neg_k, wrap, include_prob):
        self.psi_size = psi_size
        self.dt = dt
        self.time_steps = int((end_time - start_time) / dt)
        if not neg_k:
            self.K = 2 * (psi_size - 1) * np.pi / end_interval / 10
        else:
            self.K = -2 * (psi_size - 1) * np.pi / end_interval / 10

        self.mag_mat = []
        self.wrap = wrap
        self.x = np.linspace(start_interval, end_interval, psi_size)
        self.psi_values = psi_equ(self.x, self.K)
        self.psi_normalized = self.psi_values / np.linalg.norm(self.psi_values)
        self.V = potential_equ(self.x)
        self.H = self.hamiltonian(self.V)
        self.mat = self.psi_matrix(self.psi_normalized, self.H)
        self.include_prob = include_prob
        if self.include_prob:
            self.sqr_mat = self.mag_squared_matrix(self.mat)
        
       
    # Author/Source: Nobuyuki Hirakata, http://gappyfacets.com/2016/03/30/python-differentiation-matrix/
    def second_deriv_matrix(self, n):
        main_diag = np.diag(np.ones(n)) * -2
        left_off_diag = np.diag(np.ones(n - 1), k=-1) 
        right_off_diag = np.diag(np.ones(n - 1), k=1)
        M = main_diag + left_off_diag + right_off_diag
        if self.wrap:
            M[n - 1, 0] = 1
            M[0, n - 1] = 1
        return M

    def second_deriv(self, vector, x_vals):
        N = len(vector)
        dx = x_vals[1] - x_vals[0]
        return np.dot(self.second_deriv_matrix(N), vector) / dx**2

    def hamiltonian(self, v):
        v = np.diag(v)
        t = -self.H_BAR**2/(2*self.m) * self.second_deriv_matrix(self.psi_size)
        return t + v

    def psi_next(self, psi, H):
        next = ((complex(0, -1) / self.H_BAR) * np.dot(H, psi) * self.dt) + psi
        next_mag = np.linalg.norm(next)
        self.mag_mat.append(next_mag)
        return next / next_mag

    def psi_matrix(self, psi, H):
        matrix = []
        matrix.append(psi)
        for _ in range(self.time_steps - 1):
            psi_1 = self.psi_next(psi, H)
            matrix.append(psi_1)
            psi = psi_1
        return matrix

    #this is supposed to be a probability... but probability is an integral
    # see normalization condition section https://en.wikipedia.org/wiki/Wave_function#Position-space_wave_functions
    def mag_squared_matrix(self, matrix):
        return [np.absolute(vector)**2 for vector in matrix]
    
    def show_simulation(self, title, N_frames, interv, repeat, show_real, show_imag):
        y_real = []
        y_imag = []
        prob = []
        max_y = -float('inf')
        min_y = float('inf')
        
        for i in range(N_frames):
            index = int(len(self.mat)*i/float(N_frames))
            y_r = np.real(self.mat[index])
            y_i = np.imag(self.mat[index])
            y_vals = np.append(y_r, y_i)
            if self.include_prob:
                prob_vals = self.sqr_mat[index]
                prob.append(prob_vals)
                y_vals = np.append(y_vals, prob_vals)
            current_y_max = max(y_vals)
            current_y_min = min(y_vals)
            if current_y_max > max_y:
                max_y = current_y_max
            if current_y_min < min_y:
                min_y = current_y_min
            
            y_real.append(y_r)
            y_imag.append(y_i)
        
        def animate(i):
            plt.cla()
            plt.title(title)
            plt.ylim([min_y, max_y])
            plt.plot(self.x, self.V, 'k', label='V\nIteration: {}'.format(i))
            if show_real:
                plt.plot(self.x, y_real[i], label='Real')
            if show_imag:
                plt.plot(self.x, y_imag[i], label='Imaginary')
            if self.include_prob:
                plt.plot(self.x, prob[i], label='Probability')
            plt.legend()
            
        fig = plt.figure(1, figsize=(10,7))
        return ani.FuncAnimation(fig, animate, frames=N_frames, interval=interv, repeat=repeat)

# SESolver(start_interval, end_interval, start_time, end_time, psi_equ, potential_equ, potential, dt, neg_k, wrap, include_prob)
# SESover.show_simulation(title, N_frames, interv, repeat, show_real, show_imag)

# Finite Square Well
fsw_psi = lambda x, K: 1/np.sqrt(np.pi) * np.exp(-(x-np.pi)**2/(.2)) * np.exp(1j * K * x)
fsw_potential = lambda x: [0.4 if i < len(x) // 4 or i > 3 * len(x) // 4 else 0 for i in range(len(x))]
f_sqr_well = SESolver(0, 2*np.pi, 0, 1000, 501, fsw_psi, fsw_potential, 0.005, False, True, True)
fsw_sim = f_sqr_well.show_simulation('Finite Square Well', 200, 80, True, True, True)
plt.show()

# Small Bump
sb_psi = lambda x, K: 1/np.sqrt(np.pi) * np.exp(-(x-np.pi)**2/(.2)) * np.exp(1j * K * x)
sb_potential = lambda x: [0.03 if x_val > 1.0 and x_val < 1.1 else 0 for x_val in x]
small_bump = SESolver(0, 2*np.pi, 0, 800, 301, sb_psi, sb_potential, 0.005, True, True, True)
small_bump_sim = small_bump.show_simulation('Small Bump Potential', 200, 20, True, True, True)
plt.show()

# Harmonic Oscillator
ho_psi = lambda x, K: (1 / np.pi)**0.25 * np.exp(-0.5 * ((x - np.pi)**2) / 0.2)
ho_potential = lambda x: ((x - np.pi)**2) / 100
harm_osc = SESolver(0, 2*np.pi, 0, 5000, 101, ho_psi, ho_potential, 0.001, False, True, True)
harm_osc_sim = harm_osc.show_simulation('Harmonic Oscillator', 100, 100, True, True, True)
plt.show()

# Quadratic Potential
quad_psi = lambda x, K: np.exp(-0.5 * ((x - 2.5)**2) / 0.01) * np.exp(1j * np.pi / 4)
quad_potential = lambda x: (x + 2.7 - np.pi) * (x - 2.7 - np.pi) * (x - np.pi)**2 / 160 - 0.02
quad_osc = SESolver(0, 2*np.pi, 0, 500, 251, quad_psi, quad_potential, 0.001, False, True, True)
quad_sim = quad_osc.show_simulation('Quadratic Potential', 100, 150, True, True, True)
plt.show()

# Rolling Quadratic with Kinetic Energy
r_quad_psi = lambda x, K: np.exp(-0.5 * ((x - 2.5)**2) / 0.01) * np.exp(1j * np.pi * x)
r_quad_potential = lambda x: (x + 2.7 - np.pi) * (x - 2.7 - np.pi) * (x - np.pi)**2 / 160 - 0.02
# r_quad = SESolver(0, 2*np.pi, 0, 2000, 251, r_quad_psi, r_quad_potential, 0.001, False, True, True)
# r_quad_sim = r_quad.show_simulation('Rolling Quadratic with Kinetic Energy', 100, 150, True, True, True)
# plt.show()

# Delta Function Potential
# psi = lambda x, K: 1/np.sqrt(np.pi) * np.exp(-(x-np.pi)**2/(.2)) * np.exp(1j * -K * x)
# potential = lambda x: [2 if x_val > 1.0 and x_val < 1.05 else 0 for x_val in x]
# delta_function = SESolver(0, 2*np.pi, 0, 800, 301, sb_psi, sb_potential, 0.005, True, True, True)
# delta_sim = delta_func.show_simulation('Delta Function Potential', 100, 150, True, True, True)
# plt.show()


# Timestep Error Analysis
psi = lambda x, K: 1/np.sqrt(np.pi) * np.exp(-(x-np.pi)**2/(.2)) * np.exp(1j * K * x)
potential = lambda x: ((x - np.pi)**2) / 100

harm_oscs = []
dt_vals = []
N = 1
for _ in range(10):
    dt = 0.001 * N
    dt_vals.append(dt)
    harm_osc = SESolver(0, 2*np.pi, 0, 100*N, 101, psi, potential, dt, False, True, False)
    harm_oscs.append(np.array(harm_osc.mag_mat))
    N += 3

plt.figure(figsize=(12, 8))

for i in range(10):
    dt = dt_vals[i]
    harm_osc = (harm_oscs[i] - 1) * 100
    plt.plot(np.arange(0, len(harm_osc)), harm_osc, label="dt = {0:0.4f}".format(dt))

plt.title("Timestep error")
plt.xlabel("Iterations")
plt.ylabel("Percent Error")
plt.grid()
plt.legend()
plt.show()

# Wrapped Second Derivative Matrix Analysis

wp_psi = lambda x, K: 1/np.sqrt(np.pi) * np.exp(-(x-np.pi)**2/(.2)) * np.exp(1j * -K * x)
wp_potential = lambda x: [2 if x_val > 1.0 and x_val < 1.05 else 0 for x_val in x]
wave_packet_wrapped = SESolver(0, 5, 0, 500, 201, wp_psi, wp_potential, 0.001, False, True, True)
wave_packet_unwrapped = SESolver(0, 5, 0, 500, 201, wp_psi, wp_potential, 0.001, False, False, True)

matrix = wave_packet_wrapped.sqr_mat
x = np.arange(0, len(matrix))

values = []

for i in range (len(matrix)):
    vals = sum(matrix[i])
    values.append(vals)    
          
plt.figure()        
plt.plot(x, values, label = 'Wrapped')
plt.ylim([-0.9e-14+1, 0.9e-14+1])

# Unwrapped Second Derivative Matrix

matrix = wave_packet_unwrapped.sqr_mat
x = np.arange(0, len(matrix))

values1 = []

for i in range (len(matrix)):
    vals1 = sum(matrix[i])
    values1.append(vals1)    
   
values = np.array(values)
values1 = np.array(values1)
plt.plot(x, values1, label = 'Unwrapped')
plt.title('Probabilistic Errors Over Time')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()

# Percent Error
plt.figure()
percent_error = abs((values - values1))/((values + values1)/2) * 100    
plt.plot(x, percent_error)
plt.title('Wrapped V. Unwrapped')
plt.xlabel('Iterations')
plt.ylabel('Percent Error')
plt.show()
