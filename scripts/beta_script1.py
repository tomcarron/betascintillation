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

'''
Here we import and format all the experimental data
'''
array1=np.loadtxt('../Data/Bi_closed',usecols=(0))[:-5]
array2=np.loadtxt('../Data/Bi_closed',usecols=(1))[:-5]
Bi_closed=[array1,array2]

array1=np.loadtxt('../Data/Bi_open',usecols=(0))[:-5]
array2=np.loadtxt('../Data/Bi_open',usecols=(1))[:-5]
Bi_open=[array1,array2]

Bi_mod=[Bi_open[0], Bi_open[1]-Bi_closed[1]]

max_channel=np.argmax(Bi_mod[1])
print(Bi_mod[0][max_channel])
#max occurs at channel 3495, which corresponds to E=1063.66 keV. Channel 0 has 0 energy.
#Therefore each channel step represents an increase of 1063.66 /3531 keV.
Bi_peak_chan=3531
Cs_peak_chan=2292
Bi_chan_estep=1063.66 / Bi_peak_chan   #keV
Cs_chan_estep=661.7 / Cs_peak_chan
chan_step=(Bi_chan_estep+Cs_chan_estep)/2.0
Energy=Bi_mod[0] * chan_step

'''
Add error bars here. Error in energy is ~ 2 channel widths either side
'''
error_in_e=2*chan_step   #keV
print(error_in_e, 'error in energy keV')


array1=np.loadtxt('../Data/Cs_closed',usecols=(0))[:-5]
array2=np.loadtxt('../Data/Cs_closed',usecols=(1))[:-5]
Cs_closed=[array1,array2]

array1=np.loadtxt('../Data/Cs_open',usecols=(0))[:-5]
array2=np.loadtxt('../Data/Cs_open',usecols=(1))[:-5]
Cs_open=[array1,array2]

Cs_mod=[Cs_open[0], Cs_open[1]-Cs_closed[1]]
#Energy_data_array=np.array(Energy,Bi_mod[1],Cs_mod[1])
Energy_data_df=pd.DataFrame({'Energy':Energy,'Bi_counts':Bi_mod[1],'Cs_counts':Cs_mod[1]})
Energy_data_df.to_csv('Calibrated_data.csv')
'''
Import Fermi function values, plot function, find an appropriate fit
'''
fermi_df=pd.read_csv('../Data/Fermi.csv')
fermi=np.array([fermi_df['pe(m0c^2)'],fermi_df['E keV'],fermi_df['F for z=56']])


'''
Here we define all the functions needed to analyse the data, plot the Kurie plot, and switch between energy and momentum
'''
def rel_eng(Energy):
	E=1+(Energy / (0.51099895000*1000))
	return E

def spectral_intensity_distribution(Energy,counts):
	E=rel_eng(Energy)
	y=np.sqrt((counts) / (E*np.sqrt(((E**2)-1))*fermi_fit(E,popt[0],popt[1],popt[2])))
	return y

def rel_momentum(energy):
    y=np.sqrt(rel_eng(energy)**2-1)
   # y=momentum / (0.51099895000*1000)  #dimensionless
    return y

def k_momentum(momentum,counts):
	mom=rel_momentum(momentum)
	y=np.sqrt((counts) / (mom**2)*fermi_fit(mom,popt_momentum[0],popt_momentum[1],popt_momentum[2]))
	return y

def fermi_fit(x,a0,a1,a2):
	y=a0*(x**a1) + a2
	return y

def counts_from_k(rel_energy,k):
    E=e_from_rel_e(rel_energy)
    counts=(k**2)*(fermi_fit(E,popt[0],popt[1],popt[2]))*(rel_energy)*(np.sqrt((rel_energy**2) -1))
    return counts

def counts_from_k_momentum(momentum,k):
    mom=rel_momentum(momentum)
    counts=(k**2)*(fermi_fit(mom,popt_momentum[0],popt_momentum[1],popt_momentum[2]))*(momentum**2)
    return counts

def e_from_rel_e(rel_energy):
    energy=(rel_energy-1)*(0.51099895000*1000)  #in keV
    return energy

def rel_p_from_rel_e(epsilon):
    rel_p=np.sqrt(epsilon**2 -1)
    return rel_p

def p_from_rel_e(epsilon):
    p=(0.51099895000*1000)*(np.sqrt(epsilon**2 -1)) #units of keV/c
    return p
#Error formulas derived using Gaussian error propagation.

def epsilon_error(energy_error):
    epsilon_error=energy_error / (0.51099895000*1000)
    return epsilon_error

def kappa_error(epsilon_error,fermi_error,epsilon,N,F):
    count_err=count_error(N)
    N_partial=(0.5*(N**(-0.5))) * (np.sqrt((F*epsilon*np.sqrt(epsilon**2 -1))**-1))
    E_partial=(-np.sqrt(N/F)) * (((epsilon**2 - 1)+ epsilon**2)/(2*(np.sqrt(epsilon**2 -1))*(epsilon*np.sqrt(epsilon**2 -1))))
    F_partial=(-F**(-3/2)) * (np.sqrt(N/(epsilon*np.sqrt(epsilon**2 -1))))
    kappa_error=np.sqrt((count_err**2)*(N_partial**2)+(epsilon_error**2)*(E_partial**2)+(fermi_error**2)*(F_partial**2))
    return kappa_error

def count_error(N):
    error=np.sqrt(N)
    return error

'''
Here we do a fit of the fermi function for energy and momentum to use in our calculations
'''
popt, pcov = curve_fit(fermi_fit, fermi[1], fermi[2],bounds=([50,-5,0],[1e3,5,15]))
print(popt)

popt_momentum,pcov_momentum = curve_fit(fermi_fit,fermi[0], fermi[2],bounds=([-100,-5,-5],[10,5,5]))
print(popt_momentum)

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

'''
Not sure if the above model2 is correct, this peak may be IC electrons rather than a second beta transition
'''

'''
Finally we can make some plots :)
'''

plt.figure(0)
plt.plot(Bi_closed[0],Bi_closed[1])
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.title('Bi-207, closed')
plt.savefig('../plots/Bi_closed.png',dpi=400,bbox_inches='tight')

plt.figure(1)
plt.plot(Bi_open[0],Bi_open[1])
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.title('Bi-207, open')
plt.savefig('../plots/Bi_open.png',dpi=400,bbox_inches='tight')

plt.figure(2)
plt.plot(Bi_mod[0],Bi_mod[1])
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.title('Bi-207 spectrum')
plt.savefig('../plots/Bi.png',dpi=400,bbox_inches='tight')

plt.figure(3)
plt.plot(Energy,Bi_mod[1])
plt.title('Bi-207 spectrum')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.savefig('../plots/Bi_energy.png',dpi=400,bbox_inches='tight')

plt.figure(4)
plt.plot(Cs_open[0],Cs_open[1])
plt.title('Cs-137 spectrum, open')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.xlim(0,4000)
plt.savefig('../plots/Cs_open.png',dpi=400,bbox_inches='tight')


plt.figure(5)
plt.plot(Cs_closed[0],Cs_closed[1])
plt.title('Cs-137 spectrum, closed')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.xlim(0,4000)
plt.savefig('../plots/Cs_closed.png',dpi=400,bbox_inches='tight')

plt.figure(6)
plt.plot(Cs_mod[0],Cs_mod[1])
plt.title('Cs-137 spectrum')
plt.xlabel('Channel')
plt.xlim(0,4000)
plt.ylabel('Counts')
plt.savefig('../plots/Cs.png',dpi=400,bbox_inches='tight')

plt.figure(7)
plt.plot(Energy,Cs_mod[1])
plt.title('Cs-137 spectrum')
plt.xlabel('Energy (keV)')
plt.xlim(0,1000)
plt.ylabel('Counts')
plt.savefig('../plots/Cs_energy.png',dpi=400,bbox_inches='tight')



'''
Calculating error for three points at indices a,b,c=600,1000,1400
'''
delta_e=chan_step*2
fermi_error=1e-1
a,b,c=600,1000,1400
a_errorx=epsilon_error(delta_e)
a_errory=kappa_error(epsilon_error(delta_e),fermi_error,rel_eng(Energy)[a],Cs_mod[1][a],fermi_fit(rel_eng(Energy)[a],popt[0],popt[1],popt[2]))
b_errorx=epsilon_error(delta_e)
b_errory=kappa_error(epsilon_error(delta_e),fermi_error,rel_eng(Energy)[b],Cs_mod[1][b],fermi_fit(rel_eng(Energy)[b],popt[0],popt[1],popt[2]))
c_errorx=epsilon_error(delta_e)
c_errory=kappa_error(epsilon_error(delta_e),fermi_error,rel_eng(Energy)[c],Cs_mod[1][c],fermi_fit(rel_eng(Energy)[c],popt[0],popt[1],popt[2]))

plt.figure(8)
plt.plot(rel_eng(Energy),spectral_intensity_distribution(Energy,Cs_mod[1]))
plt.errorbar(rel_eng(Energy[a]),spectral_intensity_distribution(Energy[a],Cs_mod[1][a]),a_errory,a_errorx,fmt='o',capsize=5.0,ms=3.0)
plt.errorbar(rel_eng(Energy[b]),spectral_intensity_distribution(Energy[b],Cs_mod[1][b]),b_errory,b_errorx,fmt='o',capsize=5.0,ms=3.0)
plt.errorbar(rel_eng(Energy[c]),spectral_intensity_distribution(Energy[c],Cs_mod[1][c]),c_errory,c_errorx,fmt='o',capsize=5.0,ms=3.0)
plt.xlabel('$\epsilon$')
plt.ylabel('K($\epsilon$)')
plt.xlim(1.15,3)
plt.title('Kurie plot for Cs-137')
plt.savefig('../plots/Cs_Kurie.png',dpi=400,bbox_inches='tight')


plt.figure(9)
plt.scatter(rel_eng(Energy)[400:1600],spectral_intensity_distribution(Energy,Cs_mod[1])[400:1600],s=0.1)
#plt.scatter(rel_eng(Energy)[2400:2600],spectral_intensity_distribution(Energy,Cs_mod[1])[2400:2600],s=0.1)
plt.plot(rel_eng(Energy),y1,'--',label='Linear fit')
#plt.plot(rel_eng(Energy),y2,'--',label='Second transition')
plt.errorbar(rel_eng(Energy[a]),spectral_intensity_distribution(Energy[a],Cs_mod[1][a]),a_errory,a_errorx,fmt='o',capsize=5.0,ms=3.0)
plt.errorbar(rel_eng(Energy[b]),spectral_intensity_distribution(Energy[b],Cs_mod[1][b]),b_errory,b_errorx,fmt='o',capsize=5.0,ms=3.0)
plt.errorbar(rel_eng(Energy[c]),spectral_intensity_distribution(Energy[c],Cs_mod[1][c]),c_errory,c_errorx,fmt='o',capsize=5.0,ms=3.0)
plt.legend()
plt.xlabel('$\epsilon$')
plt.ylabel('K($\epsilon$)')
plt.xlim(1.15,2.25)
plt.ylim(0,10)
plt.title('Kurie plot for Cs-137.')
plt.savefig('../plots/Kurie_Cs137_fit.png',dpi=400,bbox_inches='tight')
'''
plt.figure(4)
plt.plot(rel_momentum(Energy),k_momentum(rel_momentum(Energy),Cs_mod[1]))
plt.xlabel('$\eta$')
plt.ylabel('K($\eta$)')
#plt.xlim(1.15,3)
plt.title('Kurie plot momentum for Cs-137')
plt.savefig('Kurie_Cs137_momentum.png',dpi=400,bbox_inches='tight')
'''
'''
Still need to "Add at three freely distributed point the respective statistical error according to
Gaussian error propagation.
"
'''

'''
Extrapolate the Kurie plot for the first transition to low and High B energies.
From this calculate the intensities in the respective energy regions.

i.e use the fit of the first transition to get the counts vs Energy.
'''
energy_positive=[]
for i in range(len(rel_eng(Energy))):
                 if y1[i] > 0.0:
                     energy_positive.append(rel_eng(Energy)[i])

max_energy_positive=max(energy_positive)
new_energy=np.linspace(0,max_energy_positive,1000)
new_y=model1[0]*new_energy+model1[1]
kurie_extrapolation=np.array([new_energy,new_y])

plt.figure(10)
plt.plot(e_from_rel_e(kurie_extrapolation[0]),counts_from_k(kurie_extrapolation[0],kurie_extrapolation[1]))
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.title('Extrapolated Energy spectrum of Cs-137')
plt.savefig('../plots/Extrapolated_spectrum.png',dpi=400,bbox_inches='tight')

Extrapolated_Cs_df=pd.DataFrame({'Energy (keV)':e_from_rel_e(kurie_extrapolation[0]),'Counts':counts_from_k(kurie_extrapolation[0],kurie_extrapolation[1])})
Extrapolated_Cs_df.to_csv('Extrapolated_spectrum_Cs.csv')

'''
Not sure if above plot is correct - Should it look like the fermi corrected spectrum??
Also should produce a momentum spectrum here too.
'''
'''
plt.figure(11)
plt.plot()
plt.xlabel('Momentum ')
plt.ylabel('Counts')
plt.title('Extrapolated momentum spectrum of Cs-137')
plt.savefig('../plots/Extrapolated_spectrum_momentum.png',dpi=400,bbox_inches='tight')
'''
plt.show()
