import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
#import numdifftools as ndt
import pickle
import time

def distAnce(P1,P2):
    return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)
def mInf(X):
    global V1,V2
    return (1.0/2.0)*(1.0+np.tanh((X-V1)/V2))
def lambN(X):
    global lambNbar,V3,V4
    return lambNbar*np.cosh((X-V3)/(2.0*V4))
def nInf(X):
    global V3,V4
    return (1.0/2.0)*(1.0+np.tanh((X-V3)/V4))
def dX_dt(X):
    global gL,gCa,gK
    global Vl,VCa,Vk
    global C,Iext
    retX = np.zeros(shape=X.shape)
    retX[0] = (1.0/C)*(gL*(Vl-X[0])+gCa*mInf(X[0])*(VCa-X[0])+gK*X[1]*(Vk-X[0])+Iext)
    retX[1] = lambN(X[0])*(nInf(X[0])-X[1])
    return retX
def dx(x,y):
    global Iext
    global gL,gCa,gK
    global Vl,VCa,Vk
    global C
    return (1.0/C)*(gL*(Vl-x)+gCa*mInf(x)*(VCa-x)+gK*y*(Vk-x)+Iext)
def dy(x,y):
    return lambN(x)*(nInf(x)-y)
	

def newtonRaphson(init_guess, errLim, maxIter):
    P0 = init_guess
    iteR = 0
    distA = 1.0
    while (distA>errLim) and (iteR<maxIter):
        #print P0
        #print JFunc(P0)
        #print dX_dt(P0).shape	
        P1 = P0-np.dot(np.linalg.inv(JFunc(P0)),dX_dt(P0))
        distA = distAnce(P0,P1)
        P0 = P1
        iteR += 1
    return P1		


def runSim():
    #Neuron parameters
    global gL,gCa,gK
    global Vl,VCa, Vk
    global lambNbar
    global V1,V2
    global V3,V4
    global C, varVals, init_cond
    varVals = np.zeros([nVars,runLen])
    ##Initialize
    varVals[0,0] = init_cond[0]
    varVals[1,0] = init_cond[1]
    ##Run Simulation
    for i in range(runLen-1):
        varVals[:,i+1] = varVals[:,i]+dt*dX_dt(varVals[:,i])
    return varVals

def drawVnT():
    global ax1,ax3,totTime
    varVals = runSim()
    #t = np.linspace(0,totTime,runLen)
    ax1_l1=ax1.plot(varVals[0,:],varVals[1,:])
    ax1.set_xlim([-50,65])
    ax1.set_ylim([0,1.0])
    ax3.cla()
    ax3.plot(np.linspace(0.0,totTime,len(varVals[0,:])),varVals[0,:])
    print("reset trajectories")
    ax3.set_ylim([-50,100])
    ax3.set_xlabel("time (in ms)")
    ax3.set_ylabel("voltage (in mV)")
    plt.draw()
def drawNullCline():
    global plt,ax1,x,y,ax1_l2,ax1_l3,X,Y
    dX = dx(X,Y)
    dY = dy(X,Y)
    ax1_l2=ax1.contour(X,Y,dX,levels=[0],colors='darkorange',linewidths=3)
    ax1_l3=ax1.contour(X,Y,dY,levels=[0],colors='darkgreen',linewidths=3)
    plt.draw()
def resetInitCond():
    global init_cond
    init_cond = [0.0,0.0]	
def changeRegime(label):
    global Iext, ax2, ax1
    Iext = float(label)
    A = np.loadtxt("colorMat"+str(int(float(label)))+".txt")
    ax2.cla()
    
    ax2.pcolor(A,cmap=plt.cm.Blues)
    ax2.set_xlabel("gK (in nS/cm2)")
    ax2.set_ylabel("gCa (in nS/cm2)")
    ax2.set_title(r'$Parameter\ space$')
    
    ax1.cla()
    drawNullCline()		
    drawVnT()
    ax1.set_xlabel("voltage (in mV)")
    ax1.set_ylabel("n")
    ax1.set_title(r'$Phase\ space$')

def update(val):
    global gCa, gK, ax1
    gCa = sGCa.val
    gK = sGK.val
    ax1.cla()
    drawNullCline()	
    drawVnT()
    ax1.set_xlabel("voltage (in mV)")
    ax1.set_ylabel("n")
    ax1.set_title(r'$Phase\ space$')

    
def onclick(event):
    global gCa,gK,fig,ax1,ax2,plt,x,y,init_cond
    if event.inaxes == ax1:
        init_cond = [event.xdata,event.ydata]
        ax1.cla()
        drawNullCline()		
        drawVnT()
        resetInitCond()
        ax1.set_xlabel("voltage (in mV)")
        ax1.set_ylabel("n")
        ax1.set_title(r'$Phase\ space$')


    if event.inaxes == ax2:
        gK = event.xdata
        gCa = event.ydata
        ##run The sim draw traj
        ax1.cla()
        drawNullCline()		
        drawVnT()
        ax1.set_xlabel("voltage (in mV)")
        ax1.set_ylabel("n")
        ax1.set_title(r'$Phase\ space$')

	
t1 = time.time()

global gL,gCa,gK
global Vl,VCa, Vk
global lambNbar
global V1,V2
global V3,V4
global C,JFunc
global nVars,dt,totTime
global Iext

nVars = 2
dt = 0.005
totTime = 200
runLen = int(totTime/dt) 
gL = 2.0 
gCa = 4.0 
gK = 8.0
Vl = -50.0 
VCa = 100.0 
Vk = -70.0
lambNbar = 1.0/15.0 
V1 = 10.0
V2 = 15.0 
V3 = -1.0 
V4 = 14.5
C = 20.0
Iext = 500


"""
JFunc = ndt.Jacobian(dX_dt)
diM = 20
gCas = np.linspace(0.01,20,diM) 
gKs = np.linspace(0.01,20,diM)
 
eqPts = [[[] for i in range(diM)] for j in range(diM)]
errLim = 1e-6
maxIter = 1000
init_guess = np.array([0.5,0.5])

for i in range(diM):
	#print "i is ",i
	for j in range(diM):
		#print "j is ",j
		gCa = gCas[i]
		gK = gKs[j]
		#eqPts[i][j].append((optimize.anderson(dX_dt,init_guess,iter=1000,alpha=1.0)))	
		eqPts[i][j].append(newtonRaphson(init_guess, errLim, maxIter))

                
eqPts = np.array(eqPts)
np.savetxt("eqPts.txt",eqPts,fmt="%s")
#pickle.dump(eqPts,open("eqPts.txt","wb"))


##loop over the matrix of values
##feed the eQpts to Jacobian, calculate the eigenvalues
##Decide the category it belongs to and assign corresponding color.

colorMat = np.zeros([diM,diM])

for i in range(diM):
        for j in range(diM):
                gCa = gCas[i]
                gK = gKs[j]
                eqPt =  eqPts[i][j][0]
                #print(eqPt[0],type(eqPt[0]))
                if np.isnan(eqPt[0]):
                        colorMat[i,j] = 0.0
                else:
                        #print(eqPt)
                        jac = JFunc(eqPt)
                        eigs = np.linalg.eigvals(jac)
                        #print eigs
                        p1 = eigs[0]
                        p2 = eigs[1]
                        if isinstance(p1,complex):
                                if p1.real < 0.0:
                                        colorMat[i,j] = 1.0
                                elif p1.real > 0.0:
                                        colorMat[i,j] = 2.0
                                else:
                                        colorMat[i,j] = 3.0
                        else:	
                                if p1 < 0:
                                        if p2 < 0:
                                                colorMat[i,j] = 4.0
                                        elif p2 > 0:
                                                colorMat[i,j] = 5.0
                                        else:
                                                colorMat[i,j] = 6.0
                                elif p1 > 0:
                                        if p2 < 0:
                                                colorMat[i,j] = 5.0
                                        elif p2 > 0:
                                                colorMat[i,j] = 7.0
                                        else:
                                                colorMat[i,j] = 8.0
                                else:
                                        if p2 < 0:
                                                colorMat[i,j] = 6.0
                                        elif p2 > 0:
                                                colorMat[i,j] = 8.0
                                        else:
                                                colorMat[i,j] = 9.0

np.savetxt("colorMat500.txt",colorMat)
"""
#pickle.dump(colorMat,open("colorMat400.txt","wb"))


#global varVals
#varVals = np.zeros([nVars,runLen])

global fig, ax1,ax2,ax3,x,y,X,Y,init_cond
fig = plt.figure()
fig.subplots_adjust(hspace=0.45, wspace=0.3)
ax1 = fig.add_subplot(2,3,1)
ax1.set_xlim([-60,60])
ax1.set_ylim([0.0,1.0])
ax2 = fig.add_subplot(2,3,3)
ax3 = fig.add_subplot(2,1,2)

##nullclines
init_cond = [0.0,0.0]
varVals = runSim()
ax1_l1, = ax1.plot(varVals[0,:],varVals[1,:])
x = np.linspace(-50,50,100)
y = np.linspace(0,1.0,100)
X,Y = np.meshgrid(x,y)
dX = dx(X,Y)
dY = dy(X,Y)
ax1_l2=ax1.contour(X,Y,dX,levels=[0],colors='darkorange',linewidths=3)
ax1_l3=ax1.contour(X,Y,dY,levels=[0],colors='darkgreen',linewidths=3)
ax1.set_xlabel("voltage (in mV)")
ax1.set_ylabel("n")
ax1.set_title(r'$Phase\ space$')

##colorMat
colorMat = np.loadtxt("helper_files/colorMat100.txt")
#colorMat = pickle.load(open("colorMat100.txt","rb"))
ax2.pcolor(colorMat,cmap=plt.cm.Blues)
ax3.plot(np.linspace(0.0,totTime,len(varVals[0,:])),varVals[0,:])
ax2.set_xlabel("gK (in nS/cm2)")
ax2.set_ylabel("gCa (in nS/cm2)")
ax2.set_title(r'$Parameter\ space$')

cid = fig.canvas.mpl_connect('button_press_event',onclick)

rButtonAx = plt.axes([0.45,0.7,0.1,0.2],facecolor="lightgoldenrodyellow")
radio = RadioButtons(rButtonAx, (100.0, 200.0, 300.0,500.0), active=0)
rButtonAx.set_title("Iext")

radio.on_clicked(changeRegime)

axGCa =plt.axes([0.4, 0.6, 0.13, 0.025], facecolor="red") 
axGK = plt.axes([0.4,0.55,0.13,0.025], facecolor="red")
sGCa = Slider(axGCa,"gCa",1.0,30,valinit=4.0)
sGK = Slider(axGK,"gK",0.1,30.0,valinit=8.0)
sGCa.on_changed(update)
sGK.on_changed(update)

t2 = time.time()
plt.show()

#print("The whole program took ",(t2-t1)/60.0," minutes and ",(t2-t1)%60.0," seconds")
