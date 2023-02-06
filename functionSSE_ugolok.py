import numpy as np
import matplotlib.pyplot as plt
from numba import jit

N = 5000
N_start = 100
T = 550001
delta_t = 1.0e-4
delta_x = 0.01
gamma = 5.0 / 3.0
tx = 0.847*10e-3

step1, step2, step3 = 300000, 450000, 500000 ## int( T / 100.0 ), int( T / 20.0 )

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
def functionB_rho( rho_new, r, u_new, u_old ):

    global delta_x, delta_t

    return - rho_new * ( ( r ) ** 2 * u_new - ( r - delta_x ) ** 2 * u_old ) / delta_x  / ( r ** 2 )

@jit
def functionB_U( rho, P_new, P_old ):

    global delta_x

    eps = 1.0e-9

    if ( rho <= eps ):

        return 0.0

    else:

        return - ( P_new - P_old ) / delta_x / ( rho )

@jit
def functionB_E( E_new, r, u_new, u_old ):

    global gamma, delta_x, delta_t

    return -( gamma - 1.0 ) * E_new  / ( r ** 2 ) * ( ( r ) ** 2 * u_new - ( r - delta_x ) ** 2 * u_old ) / delta_x

@jit
def functionP( rho, E ):

    global gamma

    return ( gamma - 1.0 ) * rho * E

@jit
def functionF( f_new, f_old, u_new, B ):

    global delta_x, delta_t

    return delta_t * ( B - ( u_new ) * ( f_new - f_old ) / delta_x ) + f_new #+ 0.5 * ( f_new + f_old )

@jit
def cycle_by_x( u, rho, E, P ):

    global functionB_E, functionB_rho, functionB_U
    global functionF, functionP
    global N, delta_t, delta_x

    u_new = np.zeros(N + 1)
    rho_new = np.zeros(N + 1)
    P_new = np.zeros(N + 1)
    E_new = np.zeros(N + 1)

    for j in np.arange( 1, N, 1 ):
    #while rho_new[ j ] > eps:

        rj = j * delta_x

        u_new[ j ] = functionF( u[ j ], u[ j - 1 ], u[ j ], functionB_U( rho[ j - 1 ], P[ j ], P[ j - 1 ] ) )

        rho_new[ j ] = functionF( rho[ j ], rho[ j - 1 ], u[ j ], functionB_rho( rho[ j ], rj, u[ j ], u[ j - 1 ] ) )

        E_new[ j ] = functionF( E[ j ], E[ j - 1 ], u[ j ], functionB_E( E[ j - 1 ], rj, u[ j ], u[ j - 1 ] ) )

        P_new[ j ] = functionP( rho_new[ j ], E_new[ j ] )

        rho_new[0] = rho_new[1]

        E_new[0] = E_new[1]

        P_new[0] = P_new[1]

        #print( u_new )

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