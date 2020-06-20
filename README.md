# Crystal_Bayes
The problem of finding an optimal global arrangement of Kevlar polymer chain in space (as shown in figure 1) is solved using BO technique.  The feasible set space in the present problem is define as a set of 8 tuple values {y_i,z_i }  where i=1,2,3,4. The polymer chains are allowed to displace ±0.5y_i,±0.5z_iin the y and z directions. This problem can be solved using Bayesian optimization as the objective function and the feasible set space have following properties:

	The input feasible set has dimension 8 which is less than 20 for usual BO applications
and it is a hyper-rectangle since 0≤y_i  ≤1 and 0≤z_i  ≤1, and therefore it is a simple set.
	Objective function to be optimized i.e. the potential energy surface is continuous in nature and has no known special structure i.e. linearity, concavity etc. It is very expensive to evaluate since each evaluation takes 3-6 hours and hence it is intractable to do brute force search for global optimum. 

The pseudo code for obtaining the optimal global arrangement of Kevlar using BO technique is described below:
 1. Observe n data points {(y_i,z_i ),f_i},i=1…n
 2. Build a gaussian process prior on f.
 3. Bayesian optimization
	for j=1 to n_max
		a. Obtain next (y_j,z_j ) by optimizing acquisition function EI over GP as
			(y_j,z_j )= 〖argmin〗_((y_j,z_j ) ) EI((y_j,z_j )│{(y_i,z_i ),f_i }_(i=1..n) )
		b. Obtain the ground truth f_j by running one round of black box vasp evaluation for obtaining the potential energy value.
		c. Obtain a new augmented set {(y_i,z_i ),f_i }_(i=1..n+1)
		    n = n+1
             	d. Update the gaussian process prior on f.
 
 4. Expected Improvement acquisition functionEI
     	a. (y_j,z_j )= argmin〖EI〗_n ({(y_i,z_i ),f_i }_(i=1..n))
        	Here f_i is a gaussian process prior obtained by fitting the n data points.
      	b. EI = {(μ- f^min- ξ)Φ(z)┤+ σϕ(z), 
	    where Φ and ϕ are cumulative and probability of standard normal z
              z=  (μ-f^min-ξ)/σ
