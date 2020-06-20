
# coding: utf-8

# In[3]:

import numpy as np
import os
import time 
from shutil import copyfile
import subprocess
import time
import subprocess
import time

# In[4]:

def Convert_real (HH, dr):
    #for i range(len(r)):
    pos= np.matmul(HH, dr)
    return pos
    
def Convert_scale(HHi, dr):
    pos = np.matmul(HHi, dr)
    return pos
    
def PBC_condition(rr):
    nlen = len(rr)
    #print (nlen)
    assert nlen == 3
    
    if (rr[1] < -0.5): rr[1] = rr[1] + 1
    if (rr[1] >= 0.5): rr[1] = rr[1] - 1
    if (rr[2] < -0.5): rr[2] = rr[2] + 1
    if (rr[2] >= 0.5): rr[2] = rr[2] - 1
    if (rr[0] < -0.5): rr[0] = rr[0] + 1
    if (rr[0] >= 0.5): rr[0] = rr[0] - 1
    return rr;
    
def Cutoff(ntype):
    n= (len(ntype))
    cutoff_val= np.empty([n, n])
    for i in range(n):
        for j in range(n):
            cutoff_val[i][j]= 1.5
    cutoff_val[1][1] = 1.2
    cutoff_val[1][0] = 1.4
    cutoff_val[0][1] = 1.4
    cutoff_val[1][3] = 1.2
    cutoff_val[3][1] = 1.2
    cutoff_val[1][2] = 1.2
    cutoff_val[2][1] = 1.2
    cutoff_val[2][2] = 1.2
    cutoff_val[2][1] = 1.2
    return cutoff_val


# In[15]:

class vasp:
    def __init__(self, fname= "POSCAR"):
        self.fname= fname 
        self.HH= np.zeros((3,3))
        self.atype= []
        self.ntype= []
        self.nhk=[]
        
        
    def run(self):
        self.vasp_read()
        self.Neighborlist()
        
        
    def vasp_read(self):
        self.fp = open(self.fname, 'r')
        self.fp.readline()
        self.factor=self.fp.readline().strip()
        self.HH[0]= self.fp.readline().strip().split()
        self.HH[1]= self.fp.readline().strip().split()
        self.HH[2]= self.fp.readline().strip().split()
        self.atype.extend(self.fp.readline().strip().split())
        self.ntype.extend(self.fp.readline().strip().split())
        self.ntot= np.sum(np.asarray(self.ntype, dtype=np.int))
        self.pstype= self.fp.readline()
        self.r = np.zeros((self.ntot,3))
        #self.rshift = np.zeros((self.ntot,3))

        for i in range(len(self.ntype)):
            for j in range(int(self.ntype[i])):
                self.nhk.append([i, self.atype[i]])
        

        for i in range(self.ntot):
            self.r[i]= self.fp.readline().strip().split()
    
        self.HHi= np.linalg.inv(self.HH)
        self.fp.close()
        
        #self.cutoff_val = self.cutoff_val* self.cutoff_val
        #return self.cutoff_va

    def Neighborlist (self):
        self.neighlist = []
        cutoff= Cutoff(self.ntype)
        for i in range(self.ntot):
            self.neighlist.append([])
            for j in range(self.ntot):
            #if i != j:
            #neighlist[i] = []
                dr= self.r[i]- self.r[j]
                dr= PBC_condition(dr)
                itype = self.nhk[i][0]
                jtype = self.nhk[j][0]
                dr= Convert_real(self.HH, dr)
                dr2= np.linalg.norm(dr)
                if (0.5< dr2 < cutoff[itype][jtype]):
                #print (nhk[i][1], nhk[j][1], dr2)
                    self.neighlist[i].append(j)
        #print (self.neighlist)
    def Dfs(self):
        val= (len(self.neighlist))
        self.visited = np.full((val), -1, dtype= int)
        self.idx =0
        for i in range(val):
            if (self.visited[i]==-1):
            #idx = idx+1
            #llist.append([])
                self.Dfs_search(i)
                self.idx = self.idx+1
         
        print ("Total number of cluster found:", self.idx)
        
    def Dfs_search(self, i):

        self.visited[i]=self.idx
        #print (self.visited[i])
        #llist.append(i)
        #print (visited)
        for j in self.neighlist[i]:
            if (self.visited[j] == -1):
            #visited[j]= 1
                self.Dfs_search(j)
    
    def Shift(self,infile,shiftfile, shiftval):

        shift = np.zeros((4,3))
        print(shiftval) 
        shift[0,1], shift[1,1], shift[2,1], shift[3,1] = shiftval[0], shiftval[1], shiftval[2], shiftval[3] 
        shift[0,2], shift[1,2], shift[2,2], shift[3,2] = shiftval[4], shiftval[5], shiftval[6], shiftval[7]
        shift[:,0]= 0.0
        shiftfile.write("%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n" %(shift[0,1], shift[1,1], shift[2,1], shift[3,1], shift[0,2], shift[1,2], shift[2,2], shift[3,2]))
        rshift = np.zeros((self.ntot,3))
        for i in range(self.ntot):
            rr= self.r[i]
            #print (self.nhk[i][1], Convert_real(self.HH, rr)[0], Convert_real(self.HH, rr)[1], Convert_real(self.HH, rr)[2], self.visited[i])
            if (self.visited[i]==0 or self.visited[i]==7):
                rshift[i]= rr + (0.5*shift[1])
            if (self.visited[i]==1 or self.visited[i]==4):
                rshift[i]= rr + (0.5*shift[2])
            if (self.visited[i]==5 or self.visited[i]==6):
                rshift[i]= rr + (0.5*shift[0])
            if (self.visited[i]==2 or self.visited[i]==3):
                rshift[i]= rr + (0.5*shift[3])
            #rshift[i]= np.mod(rshift[i],1)
            #print (rshift[i], rr, shift[self.visited[i]])

        fp = open(infile, 'w')
        fp.write("Kevlar \n")
        fp.write("1.000000 \n")
        fp.write((" {}  {}  {}\n").format(self.HH[0][0], self.HH[0][1], self.HH[0][2]) )
        fp.write((" {}  {}  {}\n").format(self.HH[1][0], self.HH[1][1], self.HH[1][2]) )
        fp.write((" {}  {}  {}\n").format(self.HH[2][0], self.HH[2][1], self.HH[2][2]) )
        fp.write( '    '.join( self.atype ) )
        fp.write("\n")
        fp.write( '    '.join( self.ntype ) )
        fp.write("\n")
        fp.write("Direct \n")
        print(rshift.shape)
        for i in range(self.ntot):
            fp.write(("{} {} {} \n").format(rshift[i,0], rshift[i,1], rshift[i,2]))
        fp.close()
            
    def init_setup(self,npoints,shiftval):
     
        shiftfile= open("shift.txt", 'w')
       	n = npoints 
        outdir = "iter_"+str(n)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        infile= os.path.join(outdir, "POSCAR")
        potfile= os.path.join(outdir, "POTCAR")
        kpfile= os.path.join(outdir, "KPOINTS")
        incfile= os.path.join(outdir, "INCAR")
        jobfile= os.path.join(outdir, "job.pbs")
        print (infile)
        self.Shift(infile,shiftfile, shiftval)
        copyfile("POTCAR", potfile)
        copyfile("INCAR", incfile)
        copyfile("KPOINTS", kpfile)
        copyfile("job.pbs", jobfile)
        os.chdir(outdir)
        os.system('sbatch job.pbs')
        command=["squeue" " -u" "ankitmis " "|" " grep" " big0"]

        while (True):
	    
            result=subprocess.run(command, shell=True, stdout=subprocess.PIPE)
            print(result.stdout)
            print(len(result.stdout))
            if ( len(result.stdout) == 0  ):
                break
            time.sleep(10)
        os.chdir("..")

"""

# In[17]:

v= vasp()
v.run()
v.Dfs()
v.init_setup(1,[0.1,0.2,0.3,0.4])


# In[ ]:

command=["squeue" " -u" " sctiwari" "|" " grep" " BO_Kev"]

result=subprocess.run(command, shell=True, stdout=subprocess.PIPE)
while (len(result.stdout) != 0):
    time.sleep(200)
# In[ ]:
"""


