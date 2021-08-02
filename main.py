import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial as Polyn
import math

# Parametros de ecuaciones diferenciales
beta = 0.3
gamma1 = 0.2
gamma2 = 0.2

# Parametros de optimizacion

alpha = 0.5
rho =   0.9



def derS(S,E,I,R,t):
        N = S + E + I + R
        r1 = -beta*S*I
        return r1 / float(N)

def derE(S,E,I,R,t):
        N = S + E + I + R
        r1 = -(beta*S*I) / float(N)
        return r1 - gamma1*E

def derI(S,E,I,R,t):
        return gamma1*E - gamma2*I

def derR(S,E,I,R,t):
        return gamma2*I


def derPoly(theta,t): # Derivada polinomial
	m1 = len(theta)
	pun= np.array(range(m1))
	#theta = np.dot(pun,theta)
	sum = 0
	for i in range(m1):
		t = float(t)
		sum +=  theta[i]*i*t**(float(i-1))
	return sum

def polinomio(theta,t):
	m1 = len(theta)
	sum = 0
	for i in range(m1):
		t = float(t)
		sum += theta[i]*t**(i)
	return sum

def derC(t0,t1,X,Y,T):
	sum = 0
	n = len(T)
	for idx in range(n):
		x = X[idx]
		y = Y[idx]
		t = T[idx]
		t = float(t)
		b = 1 - x- y
		d1 = derPoly(t0,t)
		d2 = derPoly(t1,t)
		f1 = d1 - function1(x,y,t)
		f2 = d2 - function2(x,y,t)
		r1 = (b+y)*(1-k)*s*(1-y)**a - x*((1-k)*(1-s)*(1-x)**a + k*(1-s)*(1-x)**a)
		r2 = (b+x)*(1-k)*(1-s)*(1-x)**a - y*((1-k)*s*(1-y)**a + k*s*(1-y)**a)
		sum += f1*r1 + f2*r2
	return -2.*sum/n

def derK(t0,t1,X,Y,T):
	sum = 0
	n = len(T)
	for idx in range(n):
		x = X[idx]
		y = Y[idx]
		t = T[idx]
		b = 1 - x -y
		d1 = derPoly(t0,t)
		d2 = derPoly(t1,t)
		f1 = d1 - function1(x,y,t)
		f2 = d2 - function2(x,y,t)
		r1 = (s-1)*(1-x)**a + (1-s)*(1-x)**a
		r1 = -(b+y)*s*(1-y)**a - x*r1 
		r1 = c*r1
		r2 = -(b+x)*(1-s)*(1-x)**a
		r2 = r2  - y*(-s*(1-y)**a + s*(1-y)**a)
		r2 = c*r2
		sum += f1*r1 + f2*r2
	return -2.*sum/n

def derS(t0,t1,X,Y,T):
	sum = 0
	n = len(T)
	for idx in range(n):
		x = X[idx]
		y = Y[idx]
		t = T[idx]
		b = 1 - x -y
		d1 = derPoly(t0,t)
		d2 = derPoly(t1,t)
		f1 = d1 - function1(x,y,t)
		f2 = d2 - function2(x,y,t)
		r1 = (1-k)*(1-x)**a - k*(1-x)**a
		r1 = (b+y)*(1-k)*(1-y)**a - x*r1
		r1 = c*r1
		r2 = (1-k)*(1-y)**a + k*(1-y)**a
		r2 = (b+x)*(1-k)*(-1)*(1-x)**a - y*r2
		r2 = c*r2
		sum += f1*r1 + f2*r2
	return -2.*sum/n

def derA(t0,t1,X,Y,T):
	sum = 0
	n = len(T)
	for idx in range(n):
		x = X[idx]
		y = Y[idx]
		t = T[idx]
		b = 1 - x - y
		d1 = derPoly(t0,t)
		d2 = derPoly(t1,t)
		f1 = d1 - function1(x,y,t)
		f2 = d2 - function2(x,y,t)
		r1 = (b+y)*(1-k)*s*(1-y)**a - x*((1-k)*(1-s)*(1-x)**a + k*(1-s)*(1-x)**a)
		r2 = (b+x)*(1-k)*(1-s)*(1-x)**a - y*((1-k)*s*(1-y)**a + k*s*(1-y)**a)
		r1 = c*r1*math.log(a)
		r2 = c*r2*math.log(a)
		sum += f1*r1 + f2*r2
	return -2.*sum/n

def Loss(th,LS,LE,LI,LR,T):
	sum = 0
	n =len(T)
	for idx in range(n):
		t = T[idx]
		S = LS[idx]
		E = LE[idx]
		I = LI[idx]
		R = LR[idx]
		#S = polinomio(t0,t)
		d1 = derPoly(th,t)
		#print("_____________")
		#print(d1)
		#print(x,y,t,a,c,k,s)
		#print(function1(x,y,t))
		#print("_____________")
		f1 = (d1 - derI(S,E,I,R,t))**2
		sum += f1 
		#print(sum)
	return sum#/float(n)


# 31/mar/19
I = [3,4,5,5,5,5,5,6,7,7,7,7,11,15,41,53,82,93,118,164,203,251,316,367,405,475,585,717,848,993,1094,1215]
T = range(len(I))

fig,ax = plt.subplots()

#ax.scatter(Bilin, color = (0.4,0.7,0.3),label='Bilingual',marker='o',s=20)
ax.scatter(T,I, color = (0,0,1),label='Infected',marker='x',s=20)
#ax.scatter(Maya, color = (1,0,0),label='Maya',marker='+',s=20)
ax.set_title(u"Infected population in Mexico")
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
ax.legend()#(loc="upper left", bbox_to_anchor=(0.8,0.2))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()

#print(Liz)

print(T)
print(Espa)

T = np.array(T)
Espa = np.array(Espa)
Maya = np.array(Maya)

"""
theta1 = Polyn.fit(T,Espa,1)#np.polyfit(T,Espa,6)
theta1 = np.array(list(theta1))

print(theta1)

print(polinomio(theta1,1920))

theta2 = Polyn.fit(T,Maya,1)#np.polyfit(T,Maya,6)
theta2 = np.array(list(theta2))
print(polinomio(theta2,1920))
print(theta2)
"""

T = T.reshape(-1,1)
Espa = Espa.reshape(-1,1)
Maya = Maya.reshape(-1,1)

poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(T)
lin2 = LinearRegression()
lin2.fit(X_poly,Espa)

poly.fit(T,Espa)

plt.scatter(T,Espa,color='red')
#plt.scatter(T,Maya)
plt.plot(T,lin2.predict(poly.fit_transform(T)),label='Spanish')
#plt.plot(T,lin3.predict(poly2.fit_transform(T)),label='Maya')
plt.legend()
plt.show()


print lin2.coef_
theta = lin2.coef_[0]
print(theta)
#print lin2.steps[1][1].coef_
theta[0] =  lin2.intercept_

#theta1 = "Inliers coef:%s - b:%0.8f" % \
#          (np.array2string(lin2.coef_,
#                           formatter={'float_kind': lambda fk: "%.8f" % fk}),
#          lin2.intercept_)
#print(theta1)
print(polinomio(theta,1920))

poly2 = PolynomialFeatures(degree=6)
X_poly2 = poly2.fit_transform(T)
lin3 = LinearRegression()
lin3.fit(X_poly2,Maya)

poly2.fit(T,Maya)

theta2 = lin3.coef_[0]
theta2[0] = lin3.intercept_

print(polinomio(theta2,1920))


plt.scatter(T,Espa,color='red')
#plt.scatter(T,Maya)
plt.plot(T,lin2.predict(poly.fit_transform(T)),label='Spanish')
#plt.plot(T,lin3.predict(poly2.fit_transform(T)),label='Maya')
plt.legend()
plt.show()


a = 1.31
c = 0.5
k = 0.5 #0.897
s = 0.5

"""
a = 1.276
c = 0.0463
k = 0.6562#0.897
s = 0.97
"""


vxc = 0
vxk = 0
vxs = 0
vxa = 0

for e in range(5000):
	vxc = rho*vxc + derC(theta,theta2,Espa,Maya,T)
	vxk = rho*vxk + derK(theta,theta2,Espa,Maya,T)
	vxs = rho*vxs + derS(theta,theta2,Espa,Maya,T)
	vxa = rho*vxa + derA(theta,theta2,Espa,Maya,T)
	caux = c - gamma*vxc
	kaux = k - gamma*vxk  #derK(theta,theta2,Espa,Maya,T)
	saux = s - gamma*vxs  #derS(theta,theta2,Espa,Maya,T)
	aaux = a - gamma*vxa  #derA(theta,theta2,Espa,Maya,T)
	c = caux + 0
	k = kaux + 0
	s = saux + 0
	a = aaux + 0
	# CORRECTION

	"""
	if a < 1.:
		a = 1.
	if a > 2.:
		a = 2.
	if c < 0.:
		c = 0.
	if c > 2.:
		c = 2.
	if k < 0.:
		k = 0.
	if k > 1.:
		k = 1.
	if s < 0.:
		s = 0.
	if s > 1.:
		s = 1.
	"""
	print(Loss(theta,theta2,Espa,Maya,T))

print(a,c,k,s)

#0.3127909578120125
#0.1142622924951695

#x0 = 0.3127909578120125
#y0 = 0.1142622924951695
x0 = polinomio(theta,1920)
y0 = polinomio(theta2,1920)


b0 = 1 - x0 - y0

print(x0,y0,b0)

h = 0.001
t = 1920

X = [x0]
Y = [y0]
B = [b0]
T1 = [t]

#print(f(0.8,0.4,0))
"""
for i in range(90000):
        #print(f(x0,y0,t))
        x1 = x0 + h*function1(x0,y0,t)
        y1 = y0 + h*function2(x0,y0,t)
        t += h
	if t == 2015 or t == 2050 or t == 2100:
		print(t,x1,y1)

        x0 = x1 + 0
        y0 = y1 + 0

        X.append(x0)
        Y.append(y0)
        B.append(1-x0-y0)
        T1.append(t)

print("DONE")

plt.plot(T1,X)
plt.scatter(T,Espa)
plt.plot(T1,Y)
plt.scatter(T,Maya)
plt.plot(T1,B)
plt.scatter(T,Bilin)
plt.show()


fig,ax = plt.subplots()

ax.scatter(T,Bilin, color = (0.4,0.7,0.3),label='Bilingual',marker='o',s=20)
ax.plot(T1,B,color = (0.4,0.7,0.3))
ax.scatter(T,Espa, color = (0,0,1),label='Spanish',marker='x',s=20)
ax.plot(T1,X,color = (0,0,1))
ax.scatter(T,Maya, color = (1,0,0),label='Maya',marker='+',s=20)
ax.plot(T1,Y,color = (1,0,0))
ax.set_title(u"Spanish-Maya interaction 1920-2015")
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
ax.legend()#(loc="upper left", bbox_to_anchor=(0.8,0.2))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()
"""
