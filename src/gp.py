""" gp.py
Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize
from vasp import *

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
	""" expected_improvement
	Expected improvement acquisition function.
	Arguments:
	----------
	x: array-like, shape = [n_samples, n_hyperparams]
		The point for which the expected improvement needs to be computed.

	gaussian_process: GaussianProcessRegressor object.
		Gaussian process trained on previously evaluated hyperparameters.

	evaluated_loss: Numpy array.
		Numpy array that contains the values off the loss function for the previously
		evaluated hyperparameters.
        
	greater_is_better: Boolean.
		Boolean flag that indicates whether the loss function is to be maximised or minimised.

	n_params: int.
		Dimension of the hyperparameter space.
	"""

	x_to_predict = x.reshape(-1, n_params)

	mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

	if greater_is_better:
		loss_optimum = np.max(evaluated_loss)
	else:
		loss_optimum = np.min(evaluated_loss)

	scaling_factor = (-1) ** (not greater_is_better)
        #epsilon = 0.1
	# In case sigma equals zero
	with np.errstate(divide='ignore'):
		Z = scaling_factor * (mu - loss_optimum) / sigma
		expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
		expected_improvement[sigma == 0.0] == 0.0

	return -1 * expected_improvement

def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
	""" 
	sample_next_hyperparameter
	Proposes the next hyperparameter to sample the loss function for.	
	Arguments:
	----------
	acquisition_func: function.
	Acquisition function to optimise.

	gaussian_process: GaussianProcessRegressor object.
		Gaussian process trained on previously evaluated hyperparameters.

	evaluated_loss: array-like, shape = [n_obs,]
		Numpy array that contains the values off the loss function for the previously
		evaluated hyperparameters.

	greater_is_better: Boolean.
		Boolean flag that indicates whether the loss function is to be maximised or minimised.

	bounds: Tuple.
		Bounds for the L-BFGS optimiser.
	n_restarts: integer.
	Number of times to run the minimiser with different starting points.
	"""
	best_x = None
	best_acquisition_value = 1
	n_params = bounds.shape[0]

	for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

		res = minimize(fun=acquisition_func,
			x0=starting_point.reshape(1, -1),
			bounds=bounds,
			method='L-BFGS-B',
			args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

		if res.fun < best_acquisition_value:
			best_acquisition_value = res.fun
			best_x = res.x

		return best_x


def bayesian_optimisation(n_iters, v, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
	""" 
	bayesian_optimisation
	Uses Gaussian Processes to optimise the loss function `sample_loss`.
	Arguments:
	----------
	n_iters: integer.
		Number of iterations to run the search algorithm.

	sample_loss: function.
		Function to be optimised.

	bounds: array-like, shape = [n_params, 2].
		Lower and upper bounds on the parameters of the function `sample_loss`.

	x0: array-like, shape = [n_pre_samples, n_params].
		Array of initial points to sample the loss function for. If None, randomly
		samples from the loss function.

	n_pre_samples: integer.
		If x0 is None, samples `n_pre_samples` initial points from the loss function.
	gp_params: dictionary.
		Dictionary of parameters to pass on to the underlying Gaussian Process.

	random_search: integer.
		Flag that indicates whether to perform random search or L-BFGS-B optimisation
		over the acquisition function.

	alpha: double.
		Variance of the error term of the GP.

	epsilon: double.
		Precision tolerance for floats.
	"""

	x_list = []
	y_list = []

	n_params = bounds.shape[0]

	if x0 is None:
		for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
			x_list.append(params)
			y_list.append(sample_loss(params))
	else:
		for params in x0:
			x_list.append(params)
		for p in sample_loss:
			y_list.append(p)

	xp = np.array(x_list)
	yp = np.array(y_list)

	# Create the GP
	if gp_params is not None:
		model = gp.GaussianProcessRegressor(**gp_params)
	else:
		kernel = gp.kernels.Matern()
		model = gp.GaussianProcessRegressor(kernel=kernel,
							alpha=alpha,
							n_restarts_optimizer=10,
							normalize_y=True)

	predictions = open('gaussian_prediction.txt', 'w')
	for n in range(n_iters):

		model.fit(xp, yp)

		# Sample next hyperparameter
		if random_search:
			x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
			ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
			next_sample = x_random[np.argmax(ei), :]
		else:
			next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=False, bounds=bounds, n_restarts=100)
		
		print(next_sample)
		
		# Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
		if np.any(np.abs(next_sample - xp) <= epsilon):
			next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

		# Sample loss for new set of parameters
		#cv_score = sample_loss
	
		fileName = 'iter_%d' %(n+13)
		
		cv_score = v.init_setup(n+13, next_sample)

		extract = open('%s/OSZICAR' %(fileName), 'r')
		lines = extract.readlines()
		line = lines[-2]
		line = line.strip().split()
		cv_score = float(line[2])
		
		#cv_score = -1485.36
		x_list.append(next_sample)
		y_list.append(cv_score)

		# Update xp and yp
		xp = np.array(x_list)
		yp = np.array(y_list)

		x = np.array(x_list[-1]).reshape(-1,8)

		mu,sigma = model.predict(x, return_std=True)
		writePred = open('%s/prediction.txt' %(fileName), 'w')
		writePred.write('mu = %12.6f, ground_truth = %12.6f,  std = %12.6f \n' %(mu, y_list[-1], sigma))
		writePred.close()


	return xp, yp, model


x0 = [[0.422787, 0.930267, 0.222655, 0.525405, 0.575388, 0.7606, 0.736996, 0.605576],
[0.404003, 0.888492, 0.277836, 0.220549, 0.854979, 0.818419, 0.238597, 0.510706],
[0.430762, 0.308705, 0.547799, 0.060303, 0.353185, 0.452617, 0.544945, 0.582585],
[0.73782, 0.389191, 0.738041, 0.341249, 0.037043, 0.572595, 0.20297, 0.939472],
[0.588046, 0.135854, 0.873878, 0.456228, 0.884405, 0.566353, 0.936225, 0.98589],
[0.019877, 0.614184, 0.192711, 0.372388, 0.029569, 0.260444, 0.380713, 0.045322],
[0.337219, 0.865705, 0.624957, 0.679925, 0.068795, 0.636234, 0.955041, 0.029574],
[0.739059, 0.765886, 0.421709, 0.233466, 0.638556, 0.612295, 0.340469, 0.28615],
[0.054939, 0.334674, 0.333284, 0.351287, 0.978352, 0.674042, 0.015841, 0.945884],
[0.808356, 0.068639, 0.108946, 0.563064, 0.377875, 0.365155, 0.25835, 0.944275],
[0.997024, 0.84939, 0.652612, 0.472219, 0.407214, 0.202641, 0.822368, 0.039624],
[0.468458, 0.176592, 0.252852, 0.240806, 0.597231, 0.270495, 0.384081, 0.046384],
[0.45048, 0.997002, 0.195098, 0.787091, 0.739537, 0.110825, 0.282915, 0.520265],
[0.582567, 0.597706, 0.787338, 0.153009, 0.684122, 0.407817, 0.201659, 0.79725],
[0.128711, 0.305412, 0.367286, 0.497277, 0.19474, 0.290487, 0.104794, 0.588482],
[0.481728, 0.066853, 0.708341, 0.772911, 0.525363, 0.014505, 0.878818, 0.111002],
[0.506304, 0.499322, 0.335009, 0.665269, 0.248572, 0.091395, 0.803199, 0.400323],
[0.629759, 0.134462, 0.768535, 0.158225, 0.982162, 0.005274, 0.669911, 0.181091],
[0.098079, 0.109494, 0.715449, 0.918005, 0.602791, 0.258974, 0.801773, 0.376486],
[0.402519, 0.764717, 0.692782, 0.998227, 0.302842, 0.945748, 0.210771, 0.559044],
[0.539983, 0.547316, 0.964631, 0.342143, 0.559458, 0.841658, 0.665438, 0.150842],
[0.286094, 0.472449, 0.71712, 0.334904, 0.524167, 0.346845, 0.41537, 0.28411],
[0.772177, 0.330249, 0.626258, 0.126384, 0.772578, 0.697005, 0.403496, 0.945846],
[0.926749, 0.279026, 0.607871, 0.951946, 0.425178, 0.295992, 0.010522, 0.023453],
[0.311697, 0.05631, 0.248306, 0.76537, 0.433359, 0.352777, 0.421645, 0.257083],
[0.186725, 0.313011, 0.671757, 0.08367, 0.578302, 0.189763, 0.059294, 0.786364],
[0.583214, 0.688566, 0.105413, 0.164368, 0.795088, 0.288322, 0.765126, 0.70733],
[0.623252, 0.661849, 0.582013, 0.935993, 0.603906, 0.684736, 0.302188, 0.551857],
[0.473207, 0.074671, 0.7819, 0.05022, 0.500008, 0.538606, 0.455922, 0.362024],
[0.024657, 0.741159, 0.342473, 0.505704, 0.174552, 0.196805, 0.870588, 0.592126],
[0.416439, 0.962213, 0.224917, 0.087395, 0.32108, 0.507907, 0.718268, 0.649242],
[0.557409, 0.729161, 0.157613, 0.376138, 0.321412, 0.580218, 0.141121, 0.880698],
[0.568098, 0.045773, 0.096111, 0.461856, 0.11848, 0.212178, 0.02144, 0.042068],
[0.768285, 0.820651, 0.103209, 0.485421, 0.174792, 0.697558, 0.096393, 0.532333],
[0.351903, 0.117403, 0.280308, 0.454899, 0.270466, 0.838374, 0.422023, 0.985692],
[0.000486, 0.128494, 0.497963, 0.982076, 0.401044, 0.236783, 0.344046, 0.135307],
[0.174035, 0.370953, 0.850171, 0.203649, 0.956653, 0.819301, 0.950161, 0.415974],
[0.966303, 0.168088, 0.591656, 0.080331, 0.3616, 0.426344, 0.652679, 0.064295],
[0.844197, 0.433022, 0.886761, 0.845594, 0.805019, 0.763101, 0.395268, 0.083112],
[0.097607, 0.841582, 0.391255, 0.011029, 0.299302, 0.954797, 0.306321, 0.996184],
[0.236795, 0.769531, 0.417868, 0.099487, 0.167284, 0.488849, 0.062958, 0.707502],
[0.945891, 0.59432, 0.158436, 0.626897, 0.270726, 0.902711, 0.647422, 0.241052],
[0.183944, 0.728173, 0.184778, 0.418688, 0.804083, 0.950984, 0.687702, 0.957508],
[0.712391, 0.777653, 0.877466, 0.771266, 0.789814, 0.138457, 0.194049, 0.489728],
[0.080581, 0.659458, 0.349908, 0.632131, 0.395104, 0.723587, 0.842672, 0.785368],
[0.546726, 0.114244, 0.314192, 0.833649, 0.160874, 0.636249, 0.325746, 0.044606],
[0.814433, 0.302234, 0.905262, 0.590904, 0.223965, 0.661966, 0.78665, 0.861987],
[0.767674, 0.979556, 0.141471, 0.293449, 0.949632, 0.024493, 0.07285, 0.576938],
[0.275583, 0.501879, 0.247898, 0.692858, 0.704069, 0.825641, 0.902381, 0.243555],
[0.218802, 0.508884, 0.708633, 0.169219, 0.416787, 0.087437, 0.459435, 0.420494],
[0.960705, 0.682529, 0.627577, 0.813043, 0.81876, 0.288116, 0.025275, 0.780055],
[0.538093, 0.257001, 0.631334, 0.046087, 0.361896, 0.231774, 0.420961, 0.799347],
[0.080953, 0.417517, 0.451131, 0.917945, 0.640913, 0.336898, 0.574038, 0.246454],
[0.534239, 0.990621, 0.933125, 0.094222, 0.461484, 0.561096, 0.77611, 0.600118],
[0.822906, 0.257745, 0.933162, 0.923515, 0.325971, 0.079582, 0.83996, 0.205246],
[0.573993, 0.758202, 0.003623, 0.03861, 0.603918, 0.089535, 0.072177, 0.728303],
[0.759133, 0.914747, 0.278633, 0.177809, 0.760901, 0.599568, 0.14633, 0.990765],
[0.382695, 0.13704, 0.027846, 0.495713, 0.465205, 0.046245, 0.534567, 0.721409],
[0.575994, 0.034412, 0.137903, 0.658704, 0.114573, 0.050102, 0.768221, 0.295732],
[0.187278, 0.336058, 0.290139, 0.666802, 0.6855, 0.024798, 0.246371, 0.127801],
[0.901852, 0.707717, 0.31251, 0.762157, 0.654437, 0.176246, 0.299161, 0.900924],
[0.789513, 0.284936, 0.853310, 0.853310, 0.938991, 0.052353, 0.046747, 0.046747],
[0.698564, 0.785632, 0.124818, 0.124818, 0.212053, 0.657131, 0.111627, 0.111627],
[0.349646, 0.156839, 0.021770, 0.021770, 0.225229, 0.991468, 0.941161, 0.941161],
[0.224974, 0.961334, 0.883803, 0.883803, 0.838012, 0.282560, 0.952491, 0.952491],
[0.181285, 0.417281, 0.284166, 0.284166, 0.762926, 0.744981, 0.552359, 0.552359],
[0.525335, 0.034845, 0.327628, 0.327628, 0.358973, 0.499951, 0.220638, 0.220638],
[0.947259, 0.121593, 0.033252, 0.033252, 0.759457, 0.153622, 0.361543, 0.361543],
[0.934054, 0.518102, 0.004903, 0.004903, 0.233488, 0.963333, 0.042538, 0.042538],
[0.266028, 0.633326, 0.769400, 0.769400, 0.354683, 0.384723, 0.073885, 0.073885],
[0.657338, 0.154531, 0.552860, 0.552860, 0.580029, 0.770080, 0.982696, 0.982696],
[0.593157, 0.583744, 0.764502, 0.764502, 0.875699, 0.175402, 0.511401, 0.511401],
[0.966064, 0.399171, 0.583609, 0.583609, 0.210634, 0.005165, 0.052853, 0.052853],
[0.730853, 0.650689, 0.846591, 0.846591, 0.459338, 0.575923, 0.443957, 0.443957]] 

#sample_loss = [-1482.0191,
sample_loss = [-1485.3623, -1483.043, -1483.6185, -1484.7854, -1486.5493, -1484.214, -1483.1003, -1485.3573, -1483.004, -1484.5708, -1486.29, -1484.2026, -1486.6741, -1484.8288, -1486.5495, -1484.6865, -1485.8311, -1486.1622, -1486.0172, -1487.0407, -1481.7036, -1483.2188, -1486.6706, -1485.3585, -1480.1183, -1484.4914, -1484.7726, -1485.2251, -1482.0391, -1486.3492, -1482.9031, -1486.2936, -1485.0585, -1486.6696, -1484.5618, -1484.0764, -1485.5571, -1486.6574, -1486.5503, -1485.9777, -1486.0139, -1487.0372, -1481.8998, -1484.6867, -1486.5461, -1487.0354, -1486.8089, -1484.5852, -1484.8312, -1483.6073, -1487.037, -1483.292, -1483.5349, -1486.5525, -1484.4925, -1485.9829, -1482.2578, -1484.6541, -1482.8535, -1487.04, -1486.2948, -1483.265560, -1484.209386, -1484.570119, -1485.116792, -1483.616461, -1483.423713, -1484.972181, -1486.016799, -1482.109280, -1484.440979, -1484.214064,-1479.258659, -1485.551543] 

bounds = [[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]
bounds = np.array(bounds)

v= vasp()
v.run()
v.Dfs()

(xp,yp,gaussian_process) = bayesian_optimisation(30, v, sample_loss, bounds, x0, n_pre_samples=74, random_search=False)
print(xp)
print('\n\n\n')
print(yp)
print('\n\n\n')
print(gaussian_process)

x_to_predict = xp.reshape(-1,8)

#x_to_predict = np.array([0.637536,0.791126,0.836563,0.654109]).reshape(-1,4)
mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

print('-------------mu------------')
print(mu)
print('------------sigma----------')
print(sigma)
print('---------- mu - sigma -----')
print(mu - sigma)
print('---------- mu + sigma -----')
print(mu + sigma)

#plt.plot(x, mu, 'o',label="GP mean")
#plt.fill_between(x, mu-sigma, mu+sigma, alpha=0.9)
#plt.show()


