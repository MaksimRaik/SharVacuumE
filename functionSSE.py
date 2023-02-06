import numpy as np
import matplotlib.pyplot as plt
from numba import jit

N = 5000
N_start = 100
T = 250001
delta_t = 1.0e-4
delta_x = 0.01
gamma = 5.0 / 3.0

step1, step2, step3 = 150000, 200000, 250000 ## int( T / 100.0 ), int( T / 20.0 )

#характерные параметры

T0 = 3500.0                               # K, Начальная температура
E0 = 3.15 * 10e9                         #удельная энергия эрг/г
rho0 = 5.45 * 10e-4                        # начальная плотность, г/см3
P0 = 1.15 * 10e6                          # Начальное давление эрг/см3
V = ( 3 * rho0 )

# Характерные параметры

Rx = 100.0                                # Rx=R0
Ux = 5.6 * 10e4                           #np.sqrt(E0/M), среднемассовая скорость
tx = 0.847*10e-3                        #Rx/Ux, sec
rhox = ( 1 / 3.0 * rho0 )
mx = ( rhox * ( Rx ** 3 ) / 3.0 )
Px = ( ( 1.0 / 3 ) * rho0 * E0 )
Ex = 3.15 * 10e9

u = np.array( [] )
rho = np.array( [] )
E = np.array( [] )
P = np.array( [] )

def start_values():

    global u, rho, P, E, N, N_start

    #начальнеые условия

    u = np.zeros( N + 1 )
    rho = np.zeros( N + 1 )
    E = np.zeros(N + 1)
    for i in np.arange( 0, N_start, 1 ):

        rho[i] = 3.0

        E[i] = 1.0

    P = rho * E * ( gamma - 1.0 )

@jit
def functionB_rho( rho_new, rho_old, r, u_new, u_old):

    global delta_x, delta_t

    return - ( rho_new + rho_old ) / 2.0 / ( r ** 2 ) * ( ( r + delta_x ) ** 2 * u_new - ( r - delta_x ) ** 2 * u_old ) / 2.0 / delta_x

@jit
def functionB_U(rho_new, rho_old, P_new, P_old):

    global delta_x

    eps = 1.0e-9

    if ( rho_new <= eps ) and ( rho_old <= eps ):

        return 0.0

    else:

        return - 2.0 * ( P_new - P_old ) / delta_x / ( rho_new + rho_old )

@jit
def functionB_E( E_new, E_old, r, u_new, u_old ):

    global gamma, delta_x, delta_t

    return -( gamma - 1.0 ) * ( E_new + E_old ) / ( r ** 2 ) * ( ( r + delta_x ) ** 2 * u_new - ( r - delta_x ) ** 2 * u_old ) / 4.0 / delta_x

@jit
def functionF( f_new, f_old, u_new, u_old, B ):

    global delta_x, delta_t

    return delta_t * ( B - ( u_new + u_old ) * ( f_new - f_old ) / 4.0 / delta_x ) + 0.5 * ( f_new + f_old )

@jit
def functionP( rho, E ):

    global gamma

    return ( gamma - 1.0 ) * rho * E

@jit
def cycle_by_x( u, rho, E, P ):

    global functionB_E, functionB_rho, functionB_U
    global functionF, functionP
    global N, delta_t, delta_x

    u_new = np.zeros(N + 1)
    rho_new = np.zeros(N + 1)
    P_new = np.zeros(N + 1)
    E_new = np.zeros(N + 1)
    #rho_new[ 1 ] = 1.0
    #j = 1
    eps = 1.0e-9

    #print( rho )

    for j in np.arange( 1, N, 1 ):
    #while rho_new[ j ] > eps:

        rj = j * delta_x

        u_new[ j ] = functionF( u[ j + 1 ], u[ j - 1 ], u[ j + 1 ], u[ j - 1 ], functionB_U( rho[ j + 1 ], rho[ j - 1 ], P[ j + 1 ], P[ j - 1 ] ) )

        rho_new[ j ] = functionF( rho[ j + 1 ], rho[ j - 1 ], u[ j + 1 ], u[ j - 1 ], functionB_rho( rho[ j + 1 ], rho[ j - 1 ], rj, u[ j + 1 ], u[ j - 1 ] ) )

        E_new[ j ] = functionF( E[ j + 1 ], E[ j - 1 ], u[ j + 1 ], u[ j - 1 ], functionB_E( E[ j + 1 ], E[ j - 1 ], rj, u[ j + 1 ], u[ j - 1 ] ) )

        P_new[ j ] = functionP( rho_new[ j ], E_new[ j ] )

        rho_new[0] = rho_new[1]
        E_new[0] = E_new[1]
        P_new[0] = P_new[1]
        #u_new[ 0 ] = u_new[ 1 ]

        #j = j + 1

    return u_new, rho_new, E_new, P_new

@jit
def cycle_by_t( u, rho, E, P ):

    global delta_t, delta_x, N, T
    global cycle_by_x

    mass_ichiout = np.zeros( ( 3, N + 1 ) )
    mass_niout = np.zeros((3, N + 1))
    mass_sanout = np.zeros((3, N + 1))

    #print( rho )

    for i in np.arange( 0, T, 1 ):

        #print( rho )

        u, rho, E, P = cycle_by_x(u, rho, E, P)

        if i == step1:

            mass_ichiout[ 0 ] = u
            mass_niout[0] = rho
            mass_sanout[0] = E

            print( "YES1" )

        elif i == step2:

            mass_ichiout[ 1 ] = u
            mass_niout[1] = rho
            mass_sanout[1] = E

            print("YES2")

        elif i == step3:

            mass_ichiout[ 2 ] = u
            mass_niout[2] = rho
            mass_sanout[2] = E

            print("YES3")

    return mass_ichiout, mass_niout, mass_sanout







