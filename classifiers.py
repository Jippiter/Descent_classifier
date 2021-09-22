# imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# prepare data
iris = genfromtxt('iris.data', delimiter='\t')
labels = iris[:, 4]
data = iris[:, :2]
# both logistic regression and svm are implemented as one-vs-one classifiers
# so reduce the three classes to two
labels = (labels != 0) * 1
# plot data distribution
col = ['b' if (i == 0) else 'r' for i in labels]
plt.scatter(data[:, 0], data[:, -1], c=col)

####Parameters to influence behaviour#####
weights_start = np.zeros(len(data[0]) + 1)
max_it = 30000
batch_size = 10
alpha_logreg = 0.001
alpha_svm = 0.0001
regularisation_svm = 10000


###Logistic regression helper functions###

# sigmoid function, takes a numpyarray and return
def sigmoid(n):
    return 1 / (1 + np.exp(-n))


# loss function, works in single element or in batch
def loss_Logistic(yi, y_hat):
    return (-yi * np.log(y_hat) - (1 - yi) * np.log(1 - y_hat))


def batch_results(subset, weights):
    result = np.zeros(subset.shape[0])
    for idx in range(result.size):
        result[idx] += np.dot(subset[idx], weights)
    return sigmoid(result)


def single_result(data, weights):
    result = 0
    result += np.dot(data, weights)
    return sigmoid(result)


def grad_LogReg(weights, y_hat, yi, xi, lamb=0.01):
    grad = np.zeros(weights.size)
    grad = weights - 2 * (y_hat - yi) * xi
    return grad


####SVM helper functions#####


def calculate_distance_batch(subset, weights, y):
    result = np.zeros(subset.shape[0])
    multipliers = np.ones(subset.shape[0])
    for idx in range(result.size):
        result[idx] = max(0, 1 - (y[idx] * np.dot(subset[idx], weights)))
        # not on support vector, multiplier of 0
        if (result[idx] == 0):
            multipliers[idx] = 0
    return result, multipliers


def loss_SVM(dist_i, weights, reg):
    # we use the hinge loss here
    loss = 1 / 2 * np.dot(weights, weights) * reg * dist_i
    return loss


def grad_SVM(weights, yi, xi, multiplier, reg):
    grad = np.zeros(weights.size)
    grad = weights - (multiplier * reg * yi) * xi
    return grad


###Big stochastic Gradient Function #####

def stochGradientDescent(input_data_x, input_data_y, gradient, loss_function, weights_k, batch_size, alpha=0.1, k=0,
                         max_it=300000, epsilon=0.000000001, svm=False, reg=0.5):
    # no limit on epochs
    indexes = np.array(range(len(input_data_x)))
    losses = np.zeros(int(max_it / 100))
    while (True):
        # shuffle index list, using this we can randomly select an x and its corresponding y
        np.random.shuffle(indexes)
        current_idx = 0
        while (current_idx <= len(indexes) - batch_size):
            # create batch from indexes
            subset = indexes[current_idx:current_idx + batch_size]
            batch_x = np.zeros((batch_size, input_data_x.shape[1]))
            batch_y = np.zeros(batch_size)
            grad_mean = np.zeros(weights_k.size)
            loss_mean = 0
            batch_idx = 0
            for idx in subset:
                batch_x[batch_idx] = input_data_x[idx]
                batch_y[batch_idx] = input_data_y[idx]
                batch_idx += 1

            # sgd routine for svm
            if svm:
                distances, multipliers = calculate_distance_batch(batch_x, weights_k, batch_y)
                # calculate gradient and loss for batch
                for idx2 in range(batch_size):
                    xi = batch_x[idx2]
                    yi = batch_y[idx2]
                    distance_i = distances[idx2]
                    multiplier = multipliers[idx2]
                    grad_mean += gradient(weights_k, yi, xi, multiplier,
                                          reg)  # calculate gradient using x&y and current weights
                    loss_mean += loss_function(distance_i, weights_k, reg)
                grad_mean = grad_mean / batch_size
                # only update loss array every 100 steps, otherwise the graph becomes too cluttered
                if (k % 100 == 0):
                    losses[int(k / 100)] = loss_mean
                weights_k1 = weights_k - (alpha * (grad_mean))
            # sgd routine for logistic regression
            else:
                # calculate y_hat for batch
                batch_y_hat = batch_results(batch_x, weights_k)

                # calculate gradient for batch
                for idx2 in range(batch_size):
                    xi = batch_x[idx2]
                    yi = batch_y[idx2]
                    y_hat = batch_y_hat[idx2]
                    # calculate gradient using x&y and current weights
                    grad_mean += gradient(weights_k, y_hat, yi, xi)  # calculate gradient using x&y and current weights
                    loss_mean += loss_function(yi, y_hat)
                grad_mean = grad_mean / batch_size
                loss_mean = loss_mean / batch_size
                # only update loss array every 100 steps, otherwise the graph becomes too cluttered
                if (k % 100 == 0):
                    losses[int(k / 100)] = loss_mean
                weights_k1 = weights_k + alpha * (grad_mean)
            # stop after predetermined amount of iterations
            if (k >= max_it - 1):
                print("stopped after k iterations")
                return weights_k1, losses
            else:
                k += 1
                weights_k = weights_k1
                # alpha = backTrack(x, gradient, alpha) Wil ik dit?
            # update index so we select the next batch
            current_idx += batch_size


# make a separate copy of the data
data_logreg = np.column_stack((np.ones(len(data)), data))
weights_end, losses = stochGradientDescent(input_data_x=data_logreg, input_data_y=labels, gradient=grad_LogReg,
                                           loss_function=loss_Logistic, weights_k=weights_start, batch_size=batch_size,
                                           max_it=max_it, alpha=alpha_logreg)
print("The final weights of the Logistic Regression classifier are:")
print(weights_end)

####Perform SVM####
# svm works with labels of 1 and -1
labels_svm = np.copy(labels)
labels_svm[labels_svm == 0] = -1
# make a separate copy of data
data_svm = np.column_stack((np.ones(len(data)), data))
weights_svm, losses_svm = stochGradientDescent(input_data_x=data_svm, input_data_y=labels_svm, gradient=grad_SVM,
                                               loss_function=loss_SVM, weights_k=weights_start, batch_size=batch_size,
                                               alpha=alpha_svm, max_it=max_it, svm=True, reg=regularisation_svm)
print("the final weights of the SVM classifier are:")
print(weights_svm)

###Plot Results Logistic Regression####
# data points
col = ['b' if (i == 0) else 'r' for i in labels]
fig1 = plt.figure(1)
ax1 = fig1.add_axes([0, 0, 1, 1])
ax1.scatter(data[:, 0], data[:, -1], c=col)
# decion boundary
x_line = np.linspace(data[:, 0].min(), data[:, 0].max(), 50)
slope_logreg = -(weights_end[1] / weights_end[2])
intercept_logreg = -(weights_end[0] / weights_end[2])
y_line_logreg = intercept_logreg + (slope_logreg * x_line)
boundary_logreg, = ax1.plot(x_line, y_line_logreg, c="k")
boundary_logreg.set_label('Decision Boundary')
ax1.legend()
fig1.suptitle('Logistic Regression Classifier')
plt.gca().axes.get_yaxis().set_visible(True)
plt.show()

# loss Logistic regression
fig2 = plt.figure(2)
x_len = np.linspace(0, int(max_it / 100), int(max_it / 100))
ax2 = fig2.add_axes([0, 0, 1, 1])
ax2.plot(x_len, losses, "b-o")
fig2.suptitle('Logistic Regression Loss')
plt.show()

####Plot results of SVM####
# datapoints
fig3 = plt.figure(3)
ax3 = fig3.add_axes([0, 0, 1, 1])
ax3.scatter(data[:, 0], data[:, -1], c=col)
# decion boundary
x_line = np.linspace(data[:, 0].min(), data[:, 0].max(), 50)
slope_svm = -(weights_svm[1] / weights_svm[2])
intercept_svm = -(weights_svm[0] / weights_svm[2])
y_line_svm = intercept_svm + (slope_svm * x_line)
boundary_svm, = ax3.plot(x_line, y_line_svm, c="k")
boundary_svm.set_label('Decision Boundary')
ax3.legend()
fig3.suptitle('SVM Classifier')
plt.show()

# Loss SVM
fig4 = plt.figure(4)
x_len_svm = np.linspace(0, int(max_it / 100), int(max_it / 100))
ax4 = fig4.add_axes([0, 0, 1, 1])
ax4.plot(x_len_svm, losses_svm, "b-o")
fig4.suptitle('SVM Loss')
plt.show()

###Plot comparison decision boundary###
fig5 = plt.figure(5)
ax5 = fig5.add_axes([0, 0, 1, 1])
ax5.scatter(data[:, 0], data[:, -1], c=col)
# decion boundary
line_logreg, = ax5.plot(x_line, y_line_logreg, c="g")
line_logreg.set_label('Logistic Regression')
line_svm, = ax5.plot(x_line, y_line_svm, c="r")
line_svm.set_label('SVM')
ax5.legend()
fig5.suptitle('Comparison of classifiers')
plt.show()
