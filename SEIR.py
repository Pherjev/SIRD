import matplotlib.pyplot as plt 
import numpy as np 
from math import floor 
import time

# PARAMETROS

beta = 0.61
gamma1 = 1/5.2
gamma2 = 0.19#0.0975
sx = 130000000

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
	

def MSE(InfR,InfP,RecR,RecP):
	# InfR: Infected Real
	# InfP: Infected Predicted
	# RecR: Recovered Real
	# RecP: Recovered Predicted
	sum = 0
	n = len(InfR)
	for t in range(n):
		#print(InfR[t],InfP[t],(InfR[t]-InfP[t])**2)
		#print(RecR)
		#print(RecR[t],RecP[t])
		#print(RecP[t])
		sum += (InfR[t]-InfP[t])**2 + (RecR[t]-RecP[t])**2
	return sum*(1./n)




# 31/mar/19
IH = [1,4,5,5,5,5,5,6,6,7,7,7,8,12,12,15,41,53,82,93,118,164,203,251,316,367,405,475,585,717,848,993,1094,1215,1378,1510,1688,1890,2143,2439,2785,3181,3441,3844,4219,4661,5014,5399,5847,6297,6875,7497,8261,8772,9501,10544,11633,12872,13842,14677,15529,16752,17799,19224,20739,22088,23471,24905,26025,27634,29616,33460,35022,36327,38324,40186,42595,45032,47144]
D  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0 ,0 ,0 ,0 ,0 ,0, 0 ,1  ,1  ,2  ,2  ,3  ,4  ,5  ,6  ,8  ,12 ,16 ,20 ,28  ,29  ,37  ,50  ,60  ,79  ,94  , 125, 141,174 ,194 ,233 ,273, 296,332  , 406, 449 ,486, 546,650,686,712,857,970,1069,1221,1305,1351,1434,1569,1732,1859,1972,2061,2154,2271,2507,2704,2961,3353,3465,3573,3926,4220,4477,4767,5045]
RH = [0,0,0,0,1,1,1,1,1,1,1,4,4,4 ,4 ,4 ,4 ,4 ,4 ,4 ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,35 ,35  ,35  ,35  ,633 ,633 ,633 , 633, 633, 633,633 ,633 ,633 ,1772,1843,1964,2125,2125,2125,2125,2627,2697,2697,2697,2697,2697,7149,8354,9086,11423,11423,11423,12377,12377,13447,13447,16810,17781,17781,20314,21824,21824,23100,25935,26990,28475,30451,31848]

print(len(IH),len(D),len(RH))

bound = 69

#IH = IH[0:bound]
#D = D[0:bound]
#RH = RH[0:bound]

z = len(IH)

# Polinomial

#RH = [0,0,0,0,1,1,1,1,1,1,1,4,4,4 ,4 ,4 ,4 ,4 ,4 ,4 ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,4  ,35 ,239  ,309  ,386  ,633 ,568 ,673 , 787, 912, 1048,1195 ,1353,1523,1772,1843]

RR = np.array(RH) + np.array(D)


print(z)

T2 = range(z)

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

	for i in range(z*10):
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
			PR.append(R1)
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

	#print(RR)
	#print(LR)

	mse = MSE(IH,PC,RR,PR)
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

        for i in range(900):
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
	print
	print
	for i in range(11):
		i += 79
		print(i,PC[i])
	#print(PI[89],PI[90],PI[91],PI[92],PI[93],PI[94],PI[95],PI[96],PI[97],PI[98],PI[99],PI[100])
	print

	return T,LE,LI,LR,LC

# GRID SEARCH

d = search1(beta,gamma1,gamma2)

eps = 0.0001
eps2 =0.0000001

for e in range(10000):
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
plt.plot(T,LE,color='yellow',label='Exposed')
plt.plot(T,LC,color='black',label='Infected')
plt.plot(T,LI,color='red',label='Infectous')
plt.scatter(T2,IH,color='black',marker='+')
plt.plot(T,LR,color='green',label='Recovered')
plt.scatter(T2,RR,color='green',marker='.')
plt.legend()
plt.show()

fig,ax = plt.subplots()

#ax.plot(T,LS, color='blue',label='Susceptible')
ax.plot(T,LE,color='yellow',label='Exposed')
ax.plot(T,LC,color='black',label='Cumulative')
ax.plot(T,LI,color='red',label='Infected')
ax.scatter(T2,IH,color='black',marker='+')
ax.plot(T,LR,color='green',label='Recovered')
ax.scatter(T2,RR,color='green',marker='.')

ax.set_xlabel("t")
#ax.set_ylabel("Casos registrados")
ax.spines['left'].set_position(('outward',10))
ax.spines['bottom'].set_position(('outward',10))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_yticks(range(10),minor=True)
#ax.set_ylim([0,100])
#ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.legend()#loc="upper left", bbox_to_anchor=(0.8,0.2))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
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
