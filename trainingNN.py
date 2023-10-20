# Modules
import torch
import numpy as np

# Function imported
from torch import optim

# Training algorithm for the Neural Network

def trainingNN(MyNN, x_train, y_train, min_err, epoch):

 # Define epsilon
 epsilon = np.sqrt(np.sqrt(np.finfo(np.float64).eps))

 # Define matrix increments
 zero_matrix = torch.zeros(x_train.size(0), x_train.size(1))
 zero_matrix[:,0] = zero_matrix[:,0] + epsilon
 epsilon_x = zero_matrix
 zero_matrix = zero_matrix * 0
 zero_matrix[:,1] = zero_matrix[:,1] + epsilon
 epsilon_y = zero_matrix

 # The input data
 idata = x_train

 # The output data (target)
 odata = y_train

 # The variables
 x = torch.reshape(idata[:,0],(x_train.size(0),1))
 y = torch.reshape(idata[:,1],(x_train.size(0),1))

 # Define the Squared Error ( (||x-y||^2) with x, y vectors,||·|| Euclidean norm)
 loss = torch.nn.MSELoss(reduction='sum')

 # Define the optimizer (lr := learning rate)
 optimizer = optim.SGD(MyNN.parameters(), lr=1e-4)

 print('\nStart training:')

 for i in range(epoch):

  # Calculate the output of the Neural Network from the given input dataset
  ynn = MyNN(idata)

  # Calculate the first derivative of the NN wrt the first input (x)
  ynn_x = (MyNN(idata + epsilon_x) - MyNN(idata)) / epsilon

  # Calculate the first derivative of the NN wrt the second input (y)
  ynn_y = (MyNN(idata + epsilon_y) - MyNN(idata)) / epsilon

  # Calculate the second derivative of the NN wrt the first input (x)
  ynn_xx = (MyNN(idata + 2*epsilon_x) - 2*MyNN(idata+epsilon_x) + MyNN(idata)) / (epsilon*epsilon)

  # Calculate the second derivative of the NN wrt the second input (y)
  ynn_yy = (MyNN(idata + 2*epsilon_y) - 2*MyNN(idata+epsilon_y) + MyNN(idata)) / (epsilon*epsilon)

  # Calculate Δu(x,y)
  # Δu(x,y) =  -2*y*(1-y)*N(x,y,p) + (1-2x)*y*(1-y)*(δN(x,y,p)/δx) +
  #          (1-2x)*y*(1-y)*(δN(x,y,p)/δx) + x*y*(1-x)*(1-y)*(δδN(x,y,p)/δx^2) +
  #          -2*x*(1-x)*N(x,y,p) + (1-2y)*x*(1-x)*(δN(x,y,p)/δy) +
  #          (1-2y)*x*(1-x)*(δN(x,y,p)/δy) + y*x*(1-y)*(1-x)*(δδN(x,y,p)/δy^2)
  u_laplacian = -2*y*(1-y)*ynn + (1-2*x)*y*(1-y)*ynn_x + \
                (1-2*x)*y*(1-y)*ynn_x + x*y*(1-x)*(1-y)*ynn_xx + \
                -2*x*(1-x)*ynn + (1-2*y)*x*(1-x)*ynn_y + \
                (1-2*y)*x*(1-x)*ynn_y + y*x*(1-y)*(1-x)*ynn_yy

  # Calculate the error between the target and the output
  error = loss(-1*u_laplacian, odata).sqrt()

  # Calculate the gradient for each tensor of weight and bias
  error.backward()  
  
  # Update the parameters
  optimizer.step()
  optimizer.zero_grad()

  # Print the error every 500 iterations
  if (i == 0):
   print("\n","The error (every 500 steps):")
  if np.mod((i+1), 500) == 1:
   print(error)
  if ((error < min_err) == True):
   return print('\nTraining has finished with',i+1,'iterations ( error =', error,')\n')

 return print('\nTraining has finished.\n')


