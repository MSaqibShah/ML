# # Notebook Imports and Packages

import matplotlib.pyplot as plt
import numpy as np

# # Example 1 : A Simple Cost Function 

# ## $f(x) = x^2+x+1$


def f(x):
    return x**2+x+1




# Generating Data
x_1 = np.linspace(start=-3, stop=3,num=500)




# Plotting

plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)

plt.xlim(-3,3)
plt.ylim(0, 8)

plt.plot(x_1,f(x_1))
plt.show()


# ## Slope and Derrivative of $f(x)$ 
# 
# ### $f(x) = x^2+x+1 $
# ### $f'(x) = 2*x+1$ 



# Slope for f(x)
def df(x):
    return 2*x+1



# Plots 
plt.figure(figsize=[15,5])

# Plot 1
plt.subplot(1,2,1)

plt.title('Cost Function', fontsize=17)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)

plt.xlim(-3,3)
plt.ylim(0, 8)

plt.plot(x_1,f(x_1), color="#ff0000", linewidth=4)

# Plot 2
plt.subplot(1,2,2)

plt.title('Slope of Cost Function', fontsize=17)
plt.xlabel('x', fontsize=14)
plt.ylabel('df(x)', fontsize=14)

plt.xlim(-3,3)
plt.ylim(-6, 8)

plt.plot(x_1,df(x_1))
z_1 = np.linspace(0,0,500)
y_1 = np.linspace(-6,8,500)

plt.grid()

plt.plot(x_1, z_1, color="#000000")
plt.plot( z_1, y_1, color="#000000")

plt.show()


# # Gradient Descent
# 
# 1. Gradient descent will initially guess a value of x that is considered lowest
# 2. It calculated the slope or gradient at that point x
# 3. then the new value of x is caculated by subtracting the slope value from the value of x
# 4. These steps are repeated untill n times to get a value of x that is actually the minimun 
#     a. This happens automatically as each time the slope or dradient will be reduced until it is alomst zero





new_x = 3
prev_x = 0 
step_multiplier = 0.1 # parameter to set how big the step we take 
precision = 0.00000001 # parameter to set how precise we want the value to be

new_x_arr = [new_x]
slope_arr = [df(new_x)]

for i in range(500):
    prev_x = new_x
    gradient = df(prev_x)
    new_x = prev_x - step_multiplier * gradient
    
    new_x_arr.append(new_x)
    slope_arr.append(df(new_x))
    
    
    # Optiminzing by setting a accepted percision 
    diff = abs(new_x-prev_x)
    if(diff <= precision):
        print("The loop ran ", i, "number of times")
        break
    

print("Local Minimiun: ", new_x)
print("Slope at local min: ", df(new_x))
print("Cost at local min", f(new_x))
    




# Plots
plt.figure(figsize=[20,5])

# Plot 1
plt.subplot(1,3,1)

plt.title('Cost Function', fontsize=17)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)

plt.xlim(-3,3)
plt.ylim(0, 8)

plt.plot(x_1,f(x_1), color="#ff0000", linewidth=4, alpha=0.7)

values = np.array(new_x_arr)
plt.scatter(new_x_arr, f(values), color="#0000FF", alpha=0.6,linewidth=7)

# Plot 2
plt.subplot(1,3,2)

plt.title('Slope of Cost Function', fontsize=17)
plt.xlabel('x', fontsize=14)
plt.ylabel('df(x)', fontsize=14)

plt.xlim(-3,3)
plt.ylim(-6, 8)

plt.plot(x_1,df(x_1), linewidth=3, alpha=0.7, color="#0000FF")
plt.scatter(new_x_arr, df(values), color="#FF0000", alpha=0.6,linewidth=7)

z_1 = np.linspace(0,0,500)
y_1 = np.linspace(-6,8,500)
plt.grid()

plt.plot(x_1, z_1, color="#000000")
plt.plot( z_1, y_1, color="#000000")

# Plot 3
plt.subplot(1,3,3)

plt.title('Slope of Cost Function (close up)', fontsize=17)
plt.xlabel('x', fontsize=14)
plt.ylabel('df(x)', fontsize=14)

plt.xlim(-0.510,-0.49)
plt.ylim(-0.025, 0.025)

plt.plot(x_1,df(x_1), linewidth=3, alpha=0.7, color="#0000FF")
plt.scatter(new_x_arr, df(values), color="#FF0000", alpha=0.6,linewidth=15)

z_1 = np.linspace(0,0,500)
y_1 = np.linspace(-6,8,500)
plt.grid()

plt.plot(x_1, z_1, color="#000000")
plt.plot( z_1, y_1, color="#000000")

plt.show()










