# Modules
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import torch
import numpy as np

# Validation algorithm for the Neural Network

def validationNN(MyNN, x_test, y_test):

 x = torch.reshape(x_test[:,0],(x_test.size(0),1))
 y = torch.reshape(x_test[:,1],(x_test.size(0),1))
 w = torch.reshape(x_test[:,2],(x_test.size(0),1))

 # Approximated u_t(x_test)
 u = MyNN(x_test[:,:2])*x*y*(1-x)*(1-y) + 0

 # Calculate l2-norm of the error, error = ( u(x,y) - u_t(x,y) ) * (w(x,y))^(1/2), summing over the all (x,y) in x_test
 error = (y_test - u.detach().numpy())*np.sqrt(w)
 l2_error = np.linalg.norm(error)

 # Plot both approximated and ideal u(x_test)
 fig = plt.figure(figsize=(9, 9))
 ax = fig.gca(projection='3d')
 ax.set_title('u(x_test) analitical solution Vs. u_t(x_test) approximated via NN', fontsize=20)
 ax.set_xlabel("x", fontsize=16)
 ax.set_ylabel("y", fontsize=16)
 ax.set_zlabel("z", fontsize=16)
 ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color='red',label='u(x_test) [ideal]',alpha=0.1)
 ax.scatter(x_test[:, 0], x_test[:, 1], u.detach().numpy(), color='green',label='u(x_test) [approximation]', alpha =0.1)
 plt.legend()
 plt.show()

 # Plot the punctual error ( u(x_test) - u_t(x_test) )
 fig_err = plt.figure(figsize=(9, 9))
 ax_err = fig_err.gca(projection='3d')
 ax_err.set_title('Punctual Error: u(x_test)-u_t(x_test)', fontsize=20)
 ax_err.set_xlabel("x", fontsize=16)
 ax_err.set_ylabel("y", fontsize=16)
 ax_err.set_zlabel("z", fontsize=16)
 ax_err.scatter(x_test[:, 0], x_test[:, 1], y_test - u.detach().numpy())
 plt.show()

 # Show both the target and the output from the trained NN
 print("\nThe output from the NN (predictions) is:\n", u, "\n\n",
       'The target output (from measurements) is:',
       "\n", y_test, "\n\n",
       'The l2-norm of the error is approximately equal to', l2_error)

 # Add toolbar to allow navigation for graphs (pan, zoom, etc...)
 root = tk.Tk()
 canvas = FigureCanvasTkAgg(fig, root)
 canvas.draw()
 canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
 toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
 toolbar.update()
 toolbar.pack(side=tk.BOTTOM, fill=tk.X)
 root_err = tk.Tk()
 canvas_err = FigureCanvasTkAgg(fig_err, root_err)
 canvas_err.draw()
 canvas_err.get_tk_widget().pack(fill=tk.BOTH, expand=True)
 toolbar_err = NavigationToolbar2Tk(canvas_err, root_err, pack_toolbar=False)
 toolbar_err.update()
 toolbar_err.pack(side=tk.BOTTOM, fill=tk.X)
 root.mainloop()
 root_err.mainloop()

