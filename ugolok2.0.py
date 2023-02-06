import matplotlib.pyplot as plt
import numpy as np
from numba import jit

spaceAccuracy = 100

hStep = 1.0 / spaceAccuracy

gamma = 5.0 / 3.0

spaceSteps = 50 * spaceAccuracy

spaceTime = 260000

tStep = 0.001

r = np.arange( 0, spaceSteps, 1 ) * hStep

u = np.zeros( ( spaceTime, spaceSteps ) )

E = np.zeros( ( spaceTime, spaceSteps ) )

rho = np.zeros( ( spaceTime, spaceSteps ) )

p = np.zeros( ( spaceTime, spaceSteps ) )

print( r )

for i in np.arange(0, spaceSteps, 1):

    if i <= spaceAccuracy:

        rho[0][i] = 1.0

        E[0][i] = 1.0 / (gamma - 1.0)

    else:

        rho[0][i] = 0.0

        E[0][i] = 0.0

p[0] = rho[0] * E[0] * (gamma - 1.0)

@jit
def Euler(r, u, rho, E, p):
    global spaceSteps, hStep, tStep, gamma, spaceTime

    for n in np.arange(0, spaceTime-1, 1):

        for m in np.arange(1, spaceSteps, 1):

            rho[n + 1][m] = rho[n][m] - tStep / hStep * (
                        (u[n][m] + u[n][m ]) / 2.0 * (rho[n][m] - rho[n][m - 1]) + rho[n][m] / (r[m] ** 2) * (
                            r[m] ** 2 * u[n][m] - r[m - 1] ** 2 * u[n][m - 1]))

            if rho[n][m - 1] <= 1.0e-9:

                u[n + 1][m] = 0.0

            else:

                u[n + 1][m] = u[n][m] - tStep / hStep * (
                            (u[n][m] + u[n][m ]) / 2.0 * (u[n][m] - u[n][m - 1]) + 1.0 / rho[n][m - 1] * (
                                p[n][m] - p[n][m - 1]))

            E[n + 1][m] = E[n][m] - tStep / hStep * (
                        (u[n][m] + u[n][m ]) / 2.0 * (E[n][m] - E[n][m - 1]) + (gamma - 1.0) / (r[m] ** 2) * E[n][
                    m - 1] * (r[m] ** 2 * u[n][m] - r[m - 1] ** 2 * u[n][m - 1]))

            p[n + 1][m] = E[n + 1][m] * rho[n + 1][m] * (gamma - 1.0)

        #u[ n + 1 ][ 0 ] = u[ n + 1 ][ 1 ]

        rho[n + 1][0] = rho[n][0]

        E[n + 1][0] = E[n ][0]

        p[n + 1][0] = p[n ][0]

        # print( u[ n ][ m ], rho[ n ][ m ], E[ n ][ m ], p[ n ][ m ] )

    return u, rho, E, p

u, rho, E, p = Euler( r, u, rho, E, p )

#print( u[ 2000 ] )

plt.figure( figsize = ( 25, 10 ) )
plt.grid()
plt.plot( r, rho[ 20000 ], r, rho[2000] )
plt.show()

plt.figure( figsize = ( 25, 10 ) )
plt.grid()
plt.plot( r, u[ 20000 ] )
plt.show()