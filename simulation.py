import numpy as np
import matplotlib.pyplot as pyp

#Defining the Images which have to be Memorized by the Network
#Zero
I_0 = np.array([[1,-1,-1,-1,-1,1],[1,-1,1,1,-1,1],[-1,-1,1,1,-1,-1],[-1,-1,1,1,-1,-1],[-1,1,1,1,1,-1],[-1,1,1,1,1,-1],[-1,-1,1,1,-1,-1],[-1,-1,1,1,-1,-1],[1,-1,1,1,-1,1],[1,-1,-1,-1,-1,1]])
I_0 = I_0.ravel()#Vectorized version of above array
#One
I_1 = np.array([[1,1,-1,-1,1,1],[1,-1,-1,-1,1,1],[-1,-1,-1,-1,1,1],[1,1,-1,-1,1,1],[1,1,-1,-1,1,1],[1,1,-1,-1,1,1],[1,1,-1,-1,1,1],[1,1,-1,-1,1,1],[1,1,-1,-1,1,1],[-1,-1,-1,-1,-1,-1]])
I_1 = I_1.ravel()#Vectorized version of above array
#Two
I_2 = np.array([[1,-1,-1,-1,-1,1],[-1,-1,1,1,-1,-1],[1,1,1,1,-1,-1],[1,1,1,1,-1,-1],[1,1,1,1,-1,-1],[1,1,1,-1,-1,1],[1,1,-1,-1,1,1],[1,-1,-1,1,1,1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1]])
I_2 = I_2.ravel()#Vectorized version of above array

#Defining the corrupted input to the Network
#Corrupted 1
I_c_1 = np.array([[1,1,1,1,-1,1],[1,-1,-1,-1,-1,1],[1,-1,-1,-1,1,-1],[1,1,1,-1,1,1],[1,1,-1,1,1,1],[1,1,-1,-1,1,1],[1,1,-1,-1,1,1],[1,1,1,-1,1,1],[1,1,-1,-1,1,1],[-1,1,-1,-1,-1,-1]])
I_c_1 = I_c_1.ravel()#Vectorized version of above array
#Corrupted 2
I_c_2 = np.array([[1,1,1,-1,-1,1],[-1,-1,1,1,-1,-1],[1,1,1,1,1,-1],[1,1,1,1,1,-1],[1,1,1,1,1,-1],[1,1,1,-1,-1,1],[1,1,1,1,1,1],[1,-1,-1,1,1,1],[-1,-1,-1,-1,-1,-1],[-1,-1,1,1,-1,-1]])
I_c_2 = I_c_2.ravel()#Vectorized version of above array

"""Code to plot the above
pyp.gray()
pyp.imshow(#Insert the 2-D array to plot)
pyp.show()
"""

#Defining the Master Differential Equation Parameters

#Number of Oscillators
N = 60

#Frequencies of the N oscillators Generated Randomly
w_c = 1e9*(np.pi)
del_w = 100e3*(2*np.pi)
w_array = np.zeros(N)
for i in range(0,N):
	#Generate the random number
	rand_num = np.random.rand(1)
	w_array[i] = w_c + del_w*rand_num[0]

#Coupling Strength
B = 4e5

#Initial Weight Matrix
I_r = I_c_2
s_initial = np.zeros((N,N))
for i in range(0,N):
	for j in range(0,N):
		s_initial[i,j] = I_r[i]*I_r[j]

#Memory Weight Matrix
s_memory = np.zeros((N,N))
p = 3.0
for i in range(0,N):
	for j in range(0,N):
		s_memory[i,j] = 1/p*(I_0[i]*I_0[j] + I_1[i]*I_1[j] + I_2[i]*I_2[j])

#Initial 'Theta' matrix(Randomized)
theta_i = np.zeros(N)
for i in range(0,N):
	#Generate the random number
	rand_num = np.random.rand(1)
	theta_i[i] = (rand_num[0] * (2*np.pi)) - (np.pi)

#Defining time parameters, step size
t_start = 0
t_stop = 50e-7
t_step = 1e-10	#Same as delta(t)
T_init = 6e-7
n_val = int((T_init - t_start)/t_step)
n_p = int((t_stop - t_start)/t_step)
#Defining the Theta Output Matrix
theta_o = np.zeros((N,n_p))
#Define the inital condition
theta_o[:,0] = theta_i

#The Master Differential Equation
#d(theta)/dt = w_n + B*sum(s_nj*sin(theta_j-theta_n))

#Solving the IVP using plain Euler Method
for i in range(0,n_val):
	for j in range(0,N):
		temp_sum = 0
		for k in range(0,N):
			temp_sum = temp_sum + s_initial[j,k]*np.sin(theta_o[k,i] - theta_o[j,i])
		theta_o[j,i+1] = theta_o[j,i] + t_step*(w_array[j] + B*temp_sum)

for i in range(n_val-1,n_p - 1):
	for j in range(0,N):
		temp_sum = 0
		for k in range(0,N):
			temp_sum = temp_sum + s_memory[j,k]*np.sin(theta_o[k,i] - theta_o[j,i])
		theta_o[j,i+1] = theta_o[j,i] + t_step*(w_array[j] + B*temp_sum)

theta_final_array = theta_o[:,n_p -1]
theta_final_diff = np.zeros(N)
for i in range(0,N):
	theta_final_diff[i] = np.cos(theta_final_array[i] - theta_final_array[0])
theta_final_image = np.reshape(theta_final_diff,(10,6))

pyp.gray()
pyp.imshow(theta_final_image)
pyp.show()





















