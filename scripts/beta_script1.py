from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import astropy.constants as const

'''
First, the conversion electrons of 207 Bi are measured, in order to relate MCA channels of the peak
with a known energy. This measurement will be used, among other things, to perform an energy
calibration during the data analysis. This spectrum peaks at 1063.66 keV.
'''

array1=np.loadtxt('../Data/Bi_closed',usecols=(0))[:-5]
array2=np.loadtxt('../Data/Bi_closed',usecols=(1))[:-5]
Bi_closed=[array1,array2]

'''
plt.figure(1)
plt.plot(Bi_closed[0],Bi_closed[1])
plt.xlabel('Channel')
plt.ylabel('Counts')
'''
array1=np.loadtxt('../Data/Bi_open',usecols=(0))[:-5]
array2=np.loadtxt('../Data/Bi_open',usecols=(1))[:-5]
Bi_open=[array1,array2]

'''
plt.figure(2)
plt.plot(Bi_open[0],Bi_open[1])
plt.xlabel('Channel')
plt.ylabel('Counts')
'''
Bi_mod=[Bi_open[0], Bi_open[1]-Bi_closed[1]]
'''
plt.figure(3)
plt.title('Bi-207 spectrum')
plt.plot(Bi_mod[0],Bi_mod[1])
plt.xlabel('Channel')
plt.ylabel('Counts')
'''
max_channel=np.argmax(Bi_mod[1])
print(Bi_mod[0][max_channel])
#max occurs at channel 3495, which corresponds to E=1063.66 keV. Channel 0 has 0 energy. 
#Therefore each channel step represents an increase of 1063.66 /3495 keV.
chan_estep=1063.66 / 3495   #keV
Energy=Bi_mod[0] * chan_estep

plt.figure(4)
plt.plot(Energy,Bi_mod[1])
plt.title('Bi-207 spectrum')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')

array1=np.loadtxt('../Data/Cs_closed',usecols=(0))[:-5]
array2=np.loadtxt('../Data/Cs_closed',usecols=(1))[:-5]
Cs_closed=[array1,array2]

array1=np.loadtxt('../Data/Cs_open',usecols=(0))[:-5]
array2=np.loadtxt('../Data/Cs_open',usecols=(1))[:-5]
Cs_open=[array1,array2]

Cs_mod=[Cs_open[0], Cs_open[1]-Cs_closed[1]]

plt.figure(5)
plt.plot(Energy,Cs_mod[1])
plt.title('Cs-137 spectrum')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')

'''
Kurie plot
k(E) vs E
'''

'''
Import Fermi function values, plot function, find an appropriate fit
'''
fermi_df=pd.read_csv('../Data/Fermi.csv')
fermi=np.array([fermi_df['pe(m0c^2)'],fermi_df['E keV'],fermi_df['F for z=56']])
print(fermi)





def fermi_fit(x,a0,a1,a2):
	y=a0*(x**a1) + a2
	return y
	
#y=fermi_fit(fermi[1],1.4,-5,1.4)    #parameters are a rough guess for fit
popt, pcov = curve_fit(fermi_fit, fermi[1], fermi[2],bounds=([50,-5,0],[1e3,5,15]))
print(popt)




plt.figure(6)
plt.scatter(fermi[1],fermi[2],marker='+')
plt.plot(fermi[1],fermi_fit(fermi[1],popt[0],popt[1],popt[2]),'r--')
plt.xlabel('E keV')
plt.ylabel('F for z=56')
plt.title('Fermi function for z=56')
plt.savefig('Fermi_func.png',dpi=400,bbox_inches='tight')

def rel_eng(Energy):
	E=1+(Energy / (0.51099895000*1000))
	return E

def spectral_intensity_distribution(Energy,counts):
	E=rel_eng(Energy)
	y=np.sqrt((counts) / (E*np.sqrt(((E**2)-1))*fermi_fit(E,popt[0],popt[1],popt[2])))
	return y
	
plt.figure(7)
plt.plot(rel_eng(Energy),spectral_intensity_distribution(Energy,Cs_mod[1]))
plt.xlabel('Energy')
plt.ylabel('K(E)')
plt.xlim(1.15,3)
plt.title('Kurie plot for Cs-137')
plt.savefig('Kurie_Cs137.png',dpi=400,bbox_inches='tight')

#print(len(rel_eng(Energy)), len(spectral_intensity_distribution(Energy,Cs_mod[1])))

Energy_trans1=rel_eng(Energy)[400:1600]
K_trans1=spectral_intensity_distribution(Energy,Cs_mod[1])[400:1600]
#Fit for first transition data
model1=np.polyfit(Energy_trans1,K_trans1,1)

Energy_trans2=rel_eng(Energy)[2400:2600]
K_trans2=spectral_intensity_distribution(Energy,Cs_mod[1])[2400:2600]
#Fit for 2nd transition data
model2=np.polyfit(Energy_trans2,K_trans2,1)
#output of model is a,b in y=ax+b
y1=model1[0]*rel_eng(Energy)+model1[1]
y2=model2[0]*rel_eng(Energy)+model2[1]

plt.figure(8)
plt.scatter(rel_eng(Energy)[400:1600],spectral_intensity_distribution(Energy,Cs_mod[1])[400:1600],s=0.1)
plt.scatter(rel_eng(Energy)[2400:2600],spectral_intensity_distribution(Energy,Cs_mod[1])[2400:2600],s=0.1)
plt.plot(rel_eng(Energy),y1,'--',label='First transition')
plt.plot(rel_eng(Energy),y2,'--',label='Second transition')
plt.legend()
plt.xlabel('Energy')
plt.ylabel('K(E)')
plt.xlim(1.15,3)
plt.ylim(0,10)
plt.title('Kurie plot for Cs-137, two competing transitions.')
plt.savefig('Kurie_Cs137_transitions.png',dpi=400,bbox_inches='tight')

'''
Still need to "Add at three freely distributed point the respective statistical error according to
Gaussian error propagation.
"
'''

plt.show()
