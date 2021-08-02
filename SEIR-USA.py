import matplotlib.pyplot as plt 
import numpy as np 
from math import floor 
import time

# PARAMETROS

beta = 0.21
gamma1 = 1/5.2
gamma2 = 0.000001#0.0975
sx = 3000

def derS(S,E,I,R,t,beta,gamma1,gamma2):
	N = S + E + I + R
	r1 = -beta*S*I
	return r1 / float(N)

def derE(S,E,I,R,t,beta,gamma1,gamma2):
	N = S + E + I + R
	r1 = (beta*S*I) / float(N)
	return r1 - gamma1*E

def derI(S,E,I,R,t,beta,gamma1,gamma2):
	return gamma1*E - gamma2*I

def derR(S,E,I,R,t,beta,gamma1,gamma2):
	return gamma2*I

def derC(S,E,I,R,t,beta,gamma1,gamma2):
	return gamma1*E
	

def MSE(InfR,InfP):
	# InfR: Infected Real
	# InfP: Infected Predicted
	sum = 0
	n = len(InfR)
	for t in range(n):
		#print(InfR[t],InfP[t],(InfR[t]-InfP[t])**2)
		sum += (InfR[t]-InfP[t])**2
	return sum*(1./n)




# 22/01/20
IH = [1,1,2,2,5,5,5,5,5,7,8,8,11,11,11,11,11,11,11,11,12,12,13,13,13,13,13,13,13,13,15,15,15,51,51,57,58,60,68,74,98,118,149,217,262,402,518,583,959,1281,1663,2179,3499,4632,6421,7783,13747,19273,25600,33276,43847,53740,65778,83836,101657,121478,140909,161837,188172,213372,243616,275586,308850,337072,366667,396223,429052]

IH40 = IH[0:50]


#R = np.array(RH) + np.array(D)


T2 = range(len(IH40))

def search1(beta,gamma1,gamma2):

	S = sx#120000000
	E = 6
	I = 3
	R = 0
	C = 0
	t = 0

	LS = [S]
	LE = [E]
	LI = [I]
	LR = [R]
	LC = [C]
	T = [t]

	h = 0.1

	PS = [S]
	PE = [E]
	PI = [I]
	PR = [R]
	PC = [C]

	for i in range(500):
        	S1 = S + h*derS(S,E,I,R,t,beta,gamma1,gamma2)
        	E1 = E + h*derE(S,E,I,R,t,beta,gamma1,gamma2)
		I1 = I + h*derI(S,E,I,R,t,beta,gamma1,gamma2)
		R1 = R + h*derR(S,E,I,R,t,beta,gamma1,gamma2)
		C1 = C + h*derC(S,E,I,R,t,beta,gamma1,gamma2)

        	t += h


		#if t < 5.:
		#	print(t,t-floor(t))

		#print(t)
	
		if t - floor(t) < h:
			#print(t)
			PC.append(C1)
			#if t < 10:
			#	print(t,I1)
        	S = S1 + 0
        	E = E1 + 0
		I = I1 + 0
		R = R1 + 0
		C = C1 + 0

        	LS.append(S)
        	LE.append(E)
		LI.append(I)
		LR.append(R)
		LC.append(C)
        	T.append(t)

	#print(len(IH))
	#print(len(PI))

	mse = MSE(IH40,PC)
	#print("MSE=" + str(mse))
	return(mse)

def search2(beta,gamma1,gamma2):

        S = sx#100000000
        E = 6
        I = 3
	C = 3
        R = 0
        t = 0

        LS = [S]
        LE = [E]
        LI = [I]
        LR = [R]
	LC = [C]
        T = [t]

        h = 0.1

        PS = [S]
        PE = [E]
        PI = [I]
        PR = [R]
	PC = [C]

        for i in range(500):
                S1 = S + h*derS(S,E,I,R,t,beta,gamma1,gamma2)
                E1 = E + h*derE(S,E,I,R,t,beta,gamma1,gamma2)
                I1 = I + h*derI(S,E,I,R,t,beta,gamma1,gamma2)
                R1 = R + h*derR(S,E,I,R,t,beta,gamma1,gamma2)
		C1 = C + h*derC(S,E,I,R,t,beta,gamma1,gamma2)

                t += h


                #if t < 5.:
                #       print(t,t-floor(t))

                #print(t)

                if t - floor(t) < h:
                        #print(t)
                        PI.append(I1)
			PC.append(C1)
                        #if t < 10:
                        #       print(t,I1)
                S = S1 + 0
                E = E1 + 0
                I = I1 + 0
                R = R1 + 0
		C = C1 + 0

                LS.append(S)
                LE.append(E)
                LI.append(I)
                LR.append(R)
		LC.append(C)
                T.append(t)

	print(PC[36])

	return T,LE,LI,LR,LC

# GRID SEARCH

d = search1(beta,gamma1,gamma2)

eps = 0.0001
eps2 =0#0.00000001

for e in range(1000): #10000
	d1 = search1(beta-eps,gamma1,gamma2-eps2)
	d2 = search1(beta,gamma1,gamma2-eps2)
	d3 = search1(beta+eps,gamma1,gamma2-eps2)
	d4 = search1(beta-eps,gamma1,gamma2)
	d5 = search1(beta+eps,gamma1,gamma2)
	d6 = search1(beta-eps,gamma1,gamma2+eps2)
	d7 = search1(beta,gamma1,gamma2+eps2)
	d8 = search1(beta+eps,gamma1,gamma2+eps2)

	D = [d1,d2,d3,d4,d5,d6,d7,d8]
	minD = min(D)
	if d <= minD:
		print("MINIMO LOCAL")
		break
	else:
		if d1 == minD:
			d = d1 + 0
			beta -= eps
			gamma2 -= eps
		elif d2 == minD:
			d = d2 + 0
			gamma2 -= eps
		elif d3 == minD:
			d = d3 + 0
			beta += eps
			gamma2 -= eps
		elif d4 == minD:
			d = d4 + 0
			beta -= eps
		elif d5 == minD:
			d = d5 + 0
			beta += eps
		elif d6 == minD:
			d = d6 + 0
			beta -= eps
			gamma2 += eps
		elif d7 == minD:
			d = d7 + 0
			gamma2 += eps
		elif d8 == minD:
			d = d8 + 0
			beta += eps
			gamma2 += eps

	print("MSE = "+str(d))

#T2 = range(len(IH))

print(beta,gamma1,gamma2)

T,LE,LI,LR,LC = search2(beta,gamma1,gamma2)

#plt.plot(T,LS,color='blue',label='Susceptible')
#plt.plot(T,LE,color='yellow',label='Exposed')
plt.plot(T,LC,color='black',label='Infected')
#plt.plot(T,LI,color='red',label='Infectous')
plt.scatter(T2,IH40,color='red',marker='+')
#plt.plot(T,LR,color='green',label='Recovered')
#plt.scatter(T2,R,color='red',label='R',marker='o')
plt.legend()
plt.show()
"""

plt.rc('font', family='Arial')


fig,ax = plt.subplots()

Espa = [.45, .48, .6147,.6493,.6687,.7041,.7199,.7153]
Bilin= [.46, .44, .3554,.3206,.3024,.2796,.2596,.271]
Maya = [.09, .08, .0299,.0301,.0289,.0163,.0205,.0137]



T2 = [1970,1980,1990,1995,2000,2005,2010,2015]


ax.plot(T,B, color = (0.4,0.7,0.3),label='Bilingual')
ax.scatter(T2,Bilin, color =(0.4,0.7,0.3),marker='o')
ax.plot(T,Y, color = (1,0,0),label='Maya')
ax.scatter(T2,Maya,color = (1,0,0),marker='+')
ax.plot(T,X, color = (0,0,1),label='Spanish')
ax.scatter(T2,Espa,color=(0,0,1),marker='x')
ax.set_title(u"Abrams-Strogatz model of Spanish-Maya interaction")
#ax.yaxis.title("t")
ax.set_xlabel("t")
#ax.set_ylabel("Casos registrados")
ax.spines['left'].set_position(('outward',10))
ax.spines['bottom'].set_position(('outward',10))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_yticks(range(10),minor=True)
#ax.set_ylim([0,100])
ax.legend(framealpha=1)
#ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.legend()#loc="upper left", bbox_to_anchor=(0.8,0.2))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()
"""
