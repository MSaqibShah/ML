#!/usr/bin/env python
# coding: utf-8

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


# # Example 2 : Multiple Minima vs Initial Guess & Advance Functions
# 
# # $$g(x) = x^4-4x^2+5$$




# Dummy Data

x_2 = np.linspace(-2,2,1000)






# Cost function
def g(x):
    return x**4-4*x*x+5

#gradient of g(x)
def dg(x):
    return 4*(x**3)-8*x
 





# Plotting 

plt.figure(figsize=[15,5])
plt.subplot(1,2,1)

# Plot 1: Cost Function g(x)
plt.xlim(-2,2)
plt.ylim(0.5,6)

plt.title("Plot 1: Cost Function g(x)", fontsize="17")
plt.xlabel("x", fontsize="14")
plt.ylabel("g(x)", fontsize="14")

plt.plot(x_2,g(x_2),  linewidth=4, alpha=0.7)

# Plot 2: Gradient or slope of Cost Function g(x)
plt.subplot(1,2,2)
plt.xlim(-2,2)
plt.ylim(-6,8)

plt.title("Plot 2: Gradient or slope of Cost Function g(x)", fontsize=17)
plt.xlabel("x", fontsize="14")
plt.ylabel("dg(x)", fontsize="14")
plt.plot(x_2, dg(x_2))

# Show Function
plt.show()


# ## Gradient Descent as a python function




def grad_desc(derr_func, initial_guess, learning_rate=0.1, precision=0.000001,max_iter=300):
    new_x = initial_guess
    prev_x = 0 
    # step_multiplier = 0.1 # parameter to set how big the step we take 
    # precision = 0.00000001 # parameter to set how precise we want the value to be
    new_x_arr = [new_x]
    slope_arr = [derr_func(new_x)]
    for i in range(max_iter):
        prev_x = new_x
        gradient = derr_func(prev_x)
        new_x = prev_x - learning_rate * gradient

        new_x_arr.append(new_x)
        slope_arr.append(derr_func(new_x))


        # Optiminzing by setting a accepted percision 
        diff = abs(new_x-prev_x)
        if(diff <= precision):
          # print("The loop ran ", i, "number of times")
            break
    return new_x, new_x_arr, slope_arr 

    # print("Local Minimiun: ", new_x)
    # print("Slope at local min: ", df(new_x))
    # print("Cost at local min", f(new_x))
    





local_min, list_x, derrv_x = grad_desc(dg,0.5)
print("No. of times the Loop Ran: ", len(list_x))
print("Local Minimum at: ", local_min)





# Plotting 

plt.figure(figsize=[15,5])
plt.subplot(1,2,1)

# Plot 1: Cost Function g(x)
plt.xlim(-2,2)
plt.ylim(0.5,6)

plt.title("Plot 1: Cost Function g(x)", fontsize="17")
plt.xlabel("x", fontsize="14")
plt.ylabel("g(x)", fontsize="14")

plt.plot(x_2,g(x_2),  linewidth=4, alpha=0.7)
list_x_a = np.array(list_x) 
plt.scatter(list_x, g(list_x_a), s=100, c="red", alpha=0.5)

# Plot 2: Gradient or slope of Cost Function g(x)
plt.subplot(1,2,2)
plt.xlim(-2,2)
plt.ylim(-6,8)

plt.title("Plot 2: Gradient of Cost Function g(x)", fontsize=17)
plt.xlabel("x", fontsize="14")
plt.ylabel("dg(x)", fontsize="14")
plt.plot(x_2, dg(x_2))

zero_axis = np.linspace(0,0,len(x_2))
y_axis = np.linspace(-7, 10, len(x_2))
plt.plot(x_2, zero_axis, c="#000")
plt.plot(zero_axis,y_axis , c="#000")
plt.scatter(list_x, dg(list_x_a), s=100, c="red",alpha=0.5)
plt.figtext(0.5, 0.01, "initial_guess=0.5, learning_rate=0.1, precision=1e-06)", ha="center", fontsize=18)

# Show Function
plt.show()





local_min, list_x, derrv_x = grad_desc(dg,-0.5)
print("No. of times the Loop Ran: ", len(list_x))
print("Local Minimum at: ", local_min)





# Plotting 

plt.figure(figsize=[15,5])
plt.subplot(1,2,1)

# Plot 1: Cost Function g(x)
plt.xlim(-2,2)
plt.ylim(0.5,6)

plt.title("Plot 1: Cost Function g(x)", fontsize="17")
plt.xlabel("x", fontsize="14")
plt.ylabel("g(x)", fontsize="14")

plt.plot(x_2,g(x_2),  linewidth=4, alpha=0.7)
list_x_a = np.array(list_x) 
plt.scatter(list_x, g(list_x_a), s=100, c="red", alpha=0.5)

# Plot 2: Gradient or slope of Cost Function g(x)
plt.subplot(1,2,2)
plt.xlim(-2,2)
plt.ylim(-6,8)

plt.title("Plot 2: Gradient of Cost Function g(x)", fontsize=17)
plt.xlabel("x", fontsize="14")
plt.ylabel("dg(x)", fontsize="14")
plt.plot(x_2, dg(x_2))

zero_axis = np.linspace(0,0,len(x_2))
y_axis = np.linspace(-7, 10, len(x_2))
plt.plot(x_2, zero_axis, c="#000")
plt.plot(zero_axis,y_axis , c="#000")
plt.scatter(list_x, dg(list_x_a), s=100, c="red",alpha=0.5)
plt.figtext(0.5, 0.01, "initial_guess=-0.5, learning_rate=0.1, precision=1e-06)", ha="center", fontsize=18)

# Show Function
plt.show()





local_min, list_x, derrv_x = grad_desc(dg,0.5, 0.2)
print("No. of times the Loop Ran: ", len(list_x))
print("Local Minimum at: ", local_min)





# Plotting 

plt.figure(figsize=[15,5])
plt.subplot(1,2,1)

# Plot 1: Cost Function g(x)
plt.xlim(-2,2)
plt.ylim(0.5,6)

plt.title("Plot 1: Cost Function g(x)", fontsize="17")
plt.xlabel("x", fontsize="14")
plt.ylabel("g(x)", fontsize="14")

plt.plot(x_2,g(x_2),  linewidth=4, alpha=0.7)
list_x_a = np.array(list_x) 
plt.scatter(list_x, g(list_x_a), s=100, c="red", alpha=0.5)

# Plot 2: Gradient or slope of Cost Function g(x)
plt.subplot(1,2,2)
plt.xlim(-2,2)
plt.ylim(-6,8)

plt.title("Plot 2: Gradient of Cost Function g(x)", fontsize=17)
plt.xlabel("x", fontsize="14")
plt.ylabel("dg(x)", fontsize="14")
plt.plot(x_2, dg(x_2))

zero_axis = np.linspace(0,0,len(x_2))
y_axis = np.linspace(-7, 10, len(x_2))
plt.plot(x_2, zero_axis, c="#000")
plt.plot(zero_axis,y_axis , c="#000")
plt.scatter(list_x, dg(list_x_a), s=100, c="red",alpha=0.5)
plt.figtext(0.5, 0.01, "initial_guess=0.5, learning_rate=0.2, precision=1e-06)", ha="center", fontsize=18)

# Show Function
plt.show()





local_min, list_x, derrv_x = grad_desc(dg,0.5, 0.2,0.01)
print("No. of times the Loop Ran: ", len(list_x))
print("Local Minimum at: ", local_min)





# Plotting 

plt.figure(figsize=[15,5])
plt.subplot(1,2,1)

# Plot 1: Cost Function g(x)
plt.xlim(-2,2)
plt.ylim(0.5,6)

plt.title("Plot 1: Cost Function g(x)", fontsize="17")
plt.xlabel("x", fontsize="14")
plt.ylabel("g(x)", fontsize="14")

plt.plot(x_2,g(x_2),  linewidth=4, alpha=0.7)
list_x_a = np.array(list_x) 
plt.scatter(list_x, g(list_x_a), s=100, c="red", alpha=0.5)

# Plot 2: Gradient or slope of Cost Function g(x)
plt.subplot(1,2,2)
plt.xlim(-2,2)
plt.ylim(-6,8)

plt.title("Plot 2: Gradient of Cost Function g(x)", fontsize=17)
plt.xlabel("x", fontsize="14")
plt.ylabel("dg(x)", fontsize="14")
plt.plot(x_2, dg(x_2))

zero_axis = np.linspace(0,0,len(x_2))
y_axis = np.linspace(-7, 10, len(x_2))
plt.plot(x_2, zero_axis, c="#000")
plt.plot(zero_axis,y_axis , c="#000")
plt.scatter(list_x, dg(list_x_a), s=100, c="red",alpha=0.5)
plt.figtext(0.5, 0.01, "initial_guess=0.5, learning_rate=0.2, precision=0.01)", ha="center", fontsize=18)

# Show Function
plt.show()


# # Example 3: Divergence and Overflow 
# ## $$h(x) = x^5-2x^4+2$$




# Dummy Data
x_3 = np.linspace(-2.5,2.5,1000)





# Cost Function
def h(x):
    return x**5-2*x**4+2
# Gradient of the functin
def dh(x):
    return 5*x**4-8*x**3





local_min, list_x, derrv_x = grad_desc(derr_func=dh,initial_guess=0.5, learning_rate=0.1,max_iter=500)
print("No. of times the Loop Ran: ", len(list_x))
print("Local Minimum at: ", local_min)





# Plotting 

plt.figure(figsize=[15,5])
plt.subplot(1,2,1)

# Plot 1: Cost Function g(x)
plt.xlim(-1.2,2.5)
plt.ylim(-1,4)

plt.title("Plot 1: Cost Function h(x)", fontsize="17")
plt.xlabel("x", fontsize="14")
plt.ylabel("h(x)", fontsize="14")

plt.plot(x_3,h(x_3),  linewidth=4, alpha=0.7)
list_x_a = np.array(list_x) 
plt.scatter(list_x, h(list_x_a), s=100, c="red", alpha=0.5)

# Plot 2: Gradient or slope of Cost Function h(x)
plt.subplot(1,2,2)
plt.xlim(-1,2)
plt.ylim(-4,5)

plt.title("Plot 2: Gradient of Cost Function h(x)", fontsize=17)
plt.xlabel("x", fontsize="14")
plt.ylabel("dh(x)", fontsize="14")
plt.plot(x_3, dh(x_3))

zero_axis = np.linspace(0,0,len(x_2))
y_axis = np.linspace(-7, 10, len(x_2))
plt.plot(x_3, zero_axis, c="#000")
plt.plot(zero_axis,y_axis , c="#000")
plt.scatter(list_x, dh(list_x_a), s=100, c="red",alpha=0.5)

# Show Function
plt.show()


# # Learning Rate and Its Effects




n = 100

# Running Gradient Descent 3 times with 3 values of gamma
low_gamma = grad_desc(dg,3,0.0005, 0.0001,n)
mid_gamma = grad_desc(dg,3,0.001, 0.0001,n)
high_gamma = grad_desc(dg,3,0.002, 0.0001,max_iter=n)

no_of_steps = list(range(0,n+1))

plt.figure(figsize=[15,5])
plt.xlim(0,n)
plt.ylim(0,50)

plt.title("Learning Rate and Step Size", fontsize="17")
plt.xlabel("No of Iteration")
plt.ylabel("g(x)")

plt.plot(no_of_steps,g(np.array(low_gamma[1])), color="green")
plt.scatter(no_of_steps,g(np.array(low_gamma[1])), s=60, color="green")

plt.plot(no_of_steps,g(np.array(mid_gamma[1])))
plt.scatter(no_of_steps,g(np.array(mid_gamma[1])), s=60)

plt.plot(no_of_steps,g(np.array(high_gamma[1])))
plt.scatter(no_of_steps,g(np.array(high_gamma[1])), s=60)








