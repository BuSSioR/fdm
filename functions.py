import numpy.matlib
import numpy.linalg as lin
from decimal import Decimal

def mass_matrix(left_width,right_width,well_width, m_b,m_w,dx):
#TODO poprawić range
    n_l=left_width/dx
    n_r=right_width/dx
    n_well=well_width/dx

    V=numpy.matlib.zeros((int(n_well+n_r+n_l),int(n_well+n_r+n_l)))
    for i in range(int(n_l)):
        V[i,i]=m_b

    for i in range (int(n_l),int(n_l+n_well)):
        V[i,i]=m_w

    for i in range(int(n_l+n_well),int((n_well+n_r+n_l))):
        V[i,i]=m_b
    
    return V


def potential_energy(left_width,right_width,well_width, barrier_left,barrier_right,dx):
    n_l=int(left_width/dx)
    n_r=int(right_width/dx)
    n_well=int(well_width/dx)

    V=numpy.matlib.zeros((int(n_well+n_r+n_l),int(n_well+n_r+n_l)))
    for i in range(n_l):
        V[i,i]=barrier_left

    for i in range(int(n_l),int(n_l+n_well)):
        V[i,i]=.0
    
    for i in range(int(n_l+n_well),int(n_well+n_r+n_l)):
        V[i,i]=barrier_right
    
    return V
    

def calc_eigs(left_width,right_width,well_width,step,h_bar,M,V):
    B=hamilton_operator(left_width,right_width,well_width,step,h_bar,M,V)
    wk,wr=lin.eigh(B)

    return wk,wr


def hamilton_operator(left_width,right_width,well_width,dx,h,M,V):
    n=int((left_width/dx+right_width/dx+well_width/dx))
    dx=dx*10**-9
    constant=-h**2/(2*dx**2)
    K=numpy.matlib.zeros((n,n))
    if numpy.size(K)==numpy.size(V):

        for i in range (1,n-1):
            K[i,i]=-0.5*constant*(1/M[i-1,i-1]+2/M[i,i]+1/M[i+1,i+1])
            K[i,i+1]=0.5*constant*(1/M[i+1,i+1]+1/M[i,i])
            K[i,i-1]=0.5*constant*(1/M[i-1,i-1]+1/M[i,i])
        ###BRZEGI###
        K[0,0]=-0.5*constant*(2/M[0,0]+1/M[1,1])
        K[0,1]=0.5*constant*(1/M[0,0]+1/M[0,0])
        K[n-1,n-1]=-0.5*constant*(1/M[n-2,n-2]+2/M[n-1,n-1])
        K[n-1,n-2]=0.5*constant*(1/M[n-1,n-1]+1/M[n-2,n-2])
        ### gdy zwykla MRS dodaje sie V w podstaci potencjalu na
        ### diagonali, w przypadku numerowa wrzuca tu sie macierz V_n ->
        ### patrz funkcja potential_energy
        H=K+V
        return H

def poisson(ro,dx,epsilon):
    dx=dx*10**-9
    three_point_2nd_derivative=numpy.matlib.zeros((numpy.shape(ro)[0],numpy.shape(ro)[0]))
    for i in range(numpy.shape(ro)[0]-1):
        three_point_2nd_derivative[i,i]=-2.0
        if i>1:
            three_point_2nd_derivative[i-1,i]=1.0
            three_point_2nd_derivative[i,i-1]=1.0

    three_point_2nd_derivative=numpy.multiply(three_point_2nd_derivative,1.0/dx)
    #phi=three_point_2nd_derivative\(-(-1.6*10**-19).*ro./epsilon).*8**16
    sec_matrix=(-(-1.6*10**-19)*numpy.true_divide(ro,epsilon)*8**16)

    phi=numpy.linalg.lstsq(three_point_2nd_derivative,sec_matrix)
    phi=numpy.subtract(phi[0],numpy.amin(phi[0]))
    phi_new=numpy.matlib.zeros((numpy.shape(ro)[0],numpy.shape(ro)[0]))
    for i in range(numpy.shape(phi)[0]-1):
        phi_new[i,i]=phi[i]

    return phi_new