
import sys

fileName = sys.argv[1]

shiftFile = open('shift.txt', 'a')

mapID = open('atomsID', 'r')
coordref = open('coords', 'r')
coords = open(fileName, 'r')
out = open('shiftVal', 'w')


atomsref = []
atoms = []
lines = coordref.readlines()

for line in lines:
	atomsref.append(list(map(float,line.strip().split()[0:3])))

print(len(atomsref))

lines = mapID.readlines()

for line in lines:
	line = line.strip().split()
	id = int(line[0])
	clusterID = int(line[1])
	atomsref[id-1].append(clusterID)

lines = coords.readlines()
counter, lineNum = 0, 0
shift = {}
shiftID = None
for line in lines:
	lineNum +=1
	if lineNum > 8:
		a = list(map(float,line.strip().split()[0:3]))
		diff = [(a[0]-atomsref[counter][0]), (a[1]-atomsref[counter][1]), (a[2]-atomsref[counter][2])]
		#print(atomsref[counter][3])
		if (atomsref[counter][3] == 0):
			shiftID = 2
		elif (atomsref[counter][3] == 1):
			shiftID = 3
		elif (atomsref[counter][3] == 2):
			shiftID = 4
		elif (atomsref[counter][3] == 5):
			shiftID = 1
		shift[shiftID] = [diff[0], 2*diff[1], 2*diff[2]]
		out.write('%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %6d\n' %(a[0],a[1],a[2], diff[0], 2*diff[1], 2*diff[2], atomsref[counter][3]))
		counter += 1



shiftFile.write('[%8.6f,%9.6f,%9.6f,%9.6f,%9.6f,%9.6f,%9.6f,%9.6f]\n' %(shift[1][1],shift[2][1],shift[3][1],shift[4][1],shift[1][2],shift[2][2],shift[3][2],shift[4][2]))





