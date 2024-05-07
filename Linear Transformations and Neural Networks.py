import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

#Transformations example, has no realation with all next functions.
def T(v):
    w = np.zeros((3,1))
    w[0,0] = 3*v[0,0]
    w[2,0] = -2*v[1,0]
    
    return w

v = np.array([[3], [5]])
w = T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

#Transformations Defined as a Matrix Multiplication, has no realation with all next functions.
def L(v):
    A = np.array([[3,0], [0,0], [0,-2]])
    print("Transformation matrix:\n", A, "\n")
    w = A @ v
    
    return w

v = np.array([[3], [5]])
w = L(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

#Can import your image as example with folloing transformation's visualization.
img = np.loadtxt('data/image.txt')
print('Shape: ',img.shape)
print(img)

#View your image
plt.scatter(img[0], img[1], s = 0.001, color = 'black')

#Horizontal Scaling (Dilation)
def T_hscaling(v):
    A = np.array([[2,0], [0,1]])
    w = A @ v
    
    return w
    
def transform_vectors(T, v1, v2):
    V = np.hstack((v1, v2))
    W = T(V)
    
    return W
    
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_hscaling = transform_vectors(T_hscaling, e1, e2)

print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)

utils.plot_transformation(T_hscaling,e1,e2)

#View your image with transformed.
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_hscaling(img)[0], T_hscaling(img)[1], s = 0.001, color = 'grey')

#Reflection about y-axis (the vertical axis)
def T_reflection_yaxis(v):
    A = np.array([[-1,0], [0,1]])
    w = A @ v
    
    return w
    
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_reflection_yaxis = transform_vectors(T_reflection_yaxis, e1, e2)

print("Original vectors:\n e1= \n", e1,"\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_reflection_yaxis)

utils.plot_transformation(T_reflection_yaxis, e1, e2)

#View your image with transformed.
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_reflection_yaxis(img)[0], T_reflection_yaxis(img)[1], s = 0.001, color = 'grey')

#Stretching by a scalar
def T_stretch(a, v):
    """
    Performs a 2D stretching transformation on a vector v using a stretching factor a.

    Args:
        a (float): The stretching factor.
        v (numpy.array): The vector (or vectors) to be stretched.

    Returns:
        numpy.array: The stretched vector.
    """

    ### START CODE HERE ###
    # Define the transformation matrix
    T = np.array([[a,0], [0,a]])
    
    # Compute the transformation
    w = T @ v
    ### END CODE HERE ###

    return w

utils.plot_transformation(lambda v: T_stretch(2, v), e1,e2)

#View your image with transformed.
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_stretch(2,img)[0], T_stretch(2,img)[1], s = 0.001, color = 'grey')

#Horizontal shear transformation
def T_hshear(m, v):
    """
    Performs a 2D horizontal shearing transformation on an array v using a shearing factor m.

    Args:
        m (float): The shearing factor.
        v (np.array): The array to be sheared.

    Returns:
        np.array: The sheared array.
    """

    ### START CODE HERE ###
    # Define the transformation matrix
    T = np.array([[1,m], [0,1]])
    
    # Compute the transformation
    w = T @ v
    
    ### END CODE HERE ###
    
    return w

utils.plot_transformation(lambda v: T_hshear(2, v), e1,e2)

#View your image with transformed.
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_hshear(2,img)[0], T_hshear(2,img)[1], s = 0.001, color = 'grey')

#Rotation matrix
def T_rotation(theta, v):
    """
    Performs a 2D rotation transformation on an array v using a rotation angle theta.

    Args:
        theta (float): The rotation angle in radians.
        v (np.array): The array to be rotated.

    Returns:
        np.array: The rotated array.
    """
    
    ### START CODE HERE ###
    # Define the transformation matrix
    T = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])
    
    # Compute the transformation
    w = T @ v
    
    ### END CODE HERE ###
    
    return w

utils.plot_transformation(lambda v: T_rotation(np.pi, v), e1,e2)

#View your image with transformed.
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_rotation(np.pi,img)[0], T_rotation(np.pi,img)[1], s = 0.001, color = 'grey')

#Composing transformations: rotation and stretch.
def T_rotation_and_stretch(theta, a, v):
    """
    Performs a combined 2D rotation and stretching transformation on an array v using a rotation angle theta and a stretching factor a.

    Args:
        theta (float): The rotation angle in radians.
        a (float): The stretching factor.
        v (np.array): The array to be transformed.

    Returns:
        np.array: The transformed array.
    """
    ### START CODE HERE ###

    rotation_T = np.array([[a,0], [0,a]])
    stretch_T = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])

    w = stretch_T @ (rotation_T @ v)

    ### END CODE HERE ###

    return w

utils.plot_transformation(lambda v: T_rotation_and_stretch(np.pi, 2, v), e1,e2)

#View your image with transformed.
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_rotation_and_stretch(np.pi,2,img)[0], T_rotation_and_stretch(np.pi,2,img)[1], s = 0.001, color = 'grey')

#Forward_propagation function.
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m), where n_x is the dimension input (in our example is 2) and m is the number of training samples
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    Y_hat -- The output of size (1, m)
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Implement Forward Propagation to calculate Z.
    ### START CODE HERE ### (~ 2 lines of code)
    Z = W @ X + b
    Y_hat = Z
    ### END CODE HERE ###
    
    return Y_hat

#parameters example
parameters = utils.initialize_parameters(2)
print(parameters)#Such as : {'W': array([[0.01671687, 0.01434513]]), 'b': array([[0.]])}

#Cost compute functions use sum of squares method.
def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

#Combine all defined functions to define a whole and simplest neural network function.
def nn_model(X, Y, num_iterations=1000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (1, number of examples)
    num_iterations -- number of iterations in the loop
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = X.shape[0]
    
    # Initialize parameters
    parameters = utils.initialize_parameters(n_x) 
    print(parameters["W"])
    # Loop
    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (~ 2 lines of code)
        # Forward propagation. Inputs: "X, parameters, n_y". Outputs: "Y_hat".
        Y_hat = parameters["W"] @ X + parameters["b"]
        
        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = np.sum((Y_hat - Y)**2)/(2*num_iterations)
        ### END CODE HERE ###
        
        
        # Parameters update.
        parameters = utils.train_nn(parameters, Y_hat, X, Y, learning_rate = 0.001) 
        
        # Print the cost every iteration.
        if print_cost:
            if i%100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

df = pd.read_csv(" ")#You can import your data here, with .csv format.

"""The .csv data file will look like this:
	x1	x2	y
0	-0.816777	-0.524620	-0.940299
1	-0.670257	1.525658	0.754702
2	-0.163599	0.268567	0.034551
3	2.141420	0.428381	1.802543
4	0.163608	0.257243	0.314790
"""

X = np.array(df[['x1','x2']]).T
Y = np.array(df['y']).reshape(1,-1)

parameters = nn_model(X,Y, num_iterations = 5000, print_cost= True)

#Do some simple predict!
def predict(X, parameters):

    W = parameters['W']
    b = parameters['b']

    Z = np.dot(W, X) + b

    return Z

y_hat = predict(X,parameters)

df['y_hat'] = y_hat[0]

for i in range(10):
    print(f"(x1,x2) = ({df.loc[i,'x1']:.2f}, {df.loc[i,'x2']:.2f}): Actual value: {df.loc[i,'y']:.2f}. Predicted value: {df.loc[i,'y_hat']:.2f}")
