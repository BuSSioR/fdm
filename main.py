from functions import mass_matrix, potential_energy,calc_eigs,poisson
import numpy as np
import matplotlib.pyplot as plt

h_bar=6.582119514*10**-16#ev/s stała diraca
epsilon=8.85*10**-12 #przenikalnosc elektryczna
left_width=30#nm szerokość lewej bariery potencjału
well_width=20#nm szerokość studni
right_width=30#nm szerokość prawej bariery potencjału
step=0.05 # krok w gridzie
numerov=0 # flaga odpowiadająca za wrzucenie do obliczeń poprawki numerova

#####GaAs####
gaas_gap=1.424#ev
m_e_gaas=0.063*0.5109989461*10**6/(299792458*299792458)
m_h_gaas=0.51*0.5109989461*10**6/(299792458*299792458)


#####InAs####
inas_gap=0.354#ev
m_e_inas=0.023*0.5109989461*10**6/(299792458*299792458)
m_h_inas=0.41*0.5109989461*10**6/(299792458*299792458)
#####Wektory mas####
M_h=mass_matrix(left_width,right_width,well_width,m_h_gaas,m_h_inas,step)
M_e=mass_matrix(left_width,right_width,well_width,m_e_gaas,m_e_inas,step)
#print(M_h)
#print(M_e)
#print(np.shape(M_h))
#print(np.shape(M_e))
###VBO####
VBO=0.5
hole_barrier=VBO*(gaas_gap-inas_gap)
electron_barrier=(1-VBO)*(gaas_gap-inas_gap)
####KONIEC STAŁE####


V_e=potential_energy(left_width,right_width,well_width,electron_barrier,electron_barrier,step)#DLA ELEKTRONÓW
V_h=potential_energy(left_width,right_width,well_width,hole_barrier,hole_barrier,step)#DLA DZIUR


wr_e,wk_e=calc_eigs(left_width,right_width,well_width,step,h_bar,M_e,V_e)#DLA ELEKTRONÓW
wr_h,wk_h=calc_eigs(left_width,right_width,well_width,step,h_bar,M_h,V_h)#DLA DZIUR



##############################################
#########OBLICZENIA SAMOUZGODNIONE############
##############################################
mix_param=0.001
V_e_mix=V_e
V_h_mix=V_h
#x=[]

#plt.plot(wk_e[:,1])
#plt.plot(wk_h[:,1])
#plt.show()
fi_e=poisson(np.power(abs(wk_e[:,0]),2),step,epsilon)
fi_h=poisson(np.power(abs(wk_h[:,0]),2),step,epsilon)

iterator=1
while(True):
    V_e_mix=V_e+fi_e#.*1.6.*10^-19.*well_width.*10^-9
    V_h_mix=V_h+fi_h#.*1.6.*10^-19.*well_width.*10^-9
    wr_e_new,wk_e_new=calc_eigs(left_width,right_width,well_width,step,h_bar,M_e,V_e_mix)#DLA ELEKTRONÓW
    wr_h_new,wk_h_new=calc_eigs(left_width,right_width,well_width,step,h_bar,M_h,V_h_mix)#DLA DZIUR

    fi_e=mix_param * fi_e + np.multiply((1-mix_param),poisson(np.power(abs(wk_e_new[:,0]),2),step,epsilon))
    fi_h=mix_param * fi_h + np.multiply((1-mix_param),poisson(np.power(abs(wk_h_new[:,0]),2),step,epsilon))
    condition=abs(wr_e_new[0]-wr_e[0])
    wk_e=wk_e_new
    wk_h=wk_h_new
    wr_e=wr_e_new
    iterator=iterator+1
    if iterator>4:
        break
    print('############')
    print('Iteracja numer: '+str(iterator))
    print('Różnica w energii 1 stanu: '+str(condition))
    print('############')