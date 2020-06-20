import numpy as np

file1 = open('shift.txt', 'r')
file2 = open('values.txt', 'w')

lines = file1.readlines()

for line in lines:
	line = line.strip().split()
	y = 0.0
	z = 0.0
	y += (3**float(line[0]) + 5**float(line[1]) + 7**float(line[2]) + 9**float(line[3]))
	z += (3**float(line[4]) + 5**float(line[5]) + 7**float(line[6]) + 9**float(line[7]))

	y = (np.log(y)/np.log(2))
	z = (np.log(z)/np.log(2))

	file2.write('%12.6f  %12.6f \n' %(y,z))
