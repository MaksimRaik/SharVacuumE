import numpy as np
import matplotlib.pyplot as plt
import functionSSE as f
import functionSSE_ugolok as fu
from timeit import *

start = default_timer()

f.start_values()

#print( f.rho )

M1_out, M2_out, M3_out = f.cycle_by_t( f.u, f.rho, f.E, f.P )

stop = default_timer()
total_time = stop - start

# output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

#out from Velocity
###############################

plt.figure( figsize = ( 18, 10 ) )
plt.grid()
plt.xlabel( ' R ' )
plt.ylabel( ' U ' )
#plt.yscale( 'log' )

number_graph = 0

for i_out in M1_out:

    number_graph += 1

    plt.plot( np.arange( f.N ) * f.delta_x, i_out[:-1] )

plt.legend( [ str( f.step1 * f.delta_t * f.tx )  + r'сек', str( f.step2 * f.delta_t * f.tx )  + r'сек', str( f.step3 * f.delta_t * f.tx)  + r'сек' ] )
plt.savefig( 'ichi_out.png', dpi = 500 )

#out from Density
###################################

plt.figure( figsize = ( 18, 10 ) )
plt.grid()
plt.xlabel( ' R ' )
plt.ylabel( r'$ \rho $' )
plt.yscale( 'log' )

number_graph = 0

for i_out in M2_out:

    number_graph += 1

    plt.plot( np.arange( f.N ) * f.delta_x, np.round( i_out[:-1], 9) )

plt.axis( [ 0, 50, 1.0e-7, 1.0e-1 ] )
plt.legend( [ str(f.step1 * f.delta_t * f.tx) + r'сек', str( f.step2 * f.delta_t * f.tx ) + r'сек', str( f.step3 * f.delta_t  * f.tx ) + r'сек' ] )
#plt.show()
plt.savefig( 'ni_out.png', dpi = 500 )

#out from Energy
#############################

plt.figure( figsize = ( 18, 10 ) )
plt.grid()
plt.xlabel( ' R ' )
plt.ylabel( ' E ' )
plt.yscale( 'log' )

number_graph = 0

for i_out in M3_out:

    number_graph += 1

    plt.plot( np.arange( f.N ) * f.delta_x, np.round( i_out[:-1], 9) )

plt.axis( [ 0, 50, 1.0e-7, 1.0e-1 ] )
plt.legend( [ str( f.step1 * f.delta_t * f.tx )  + r'сек', str( f.step2 * f.delta_t * f.tx )  + r'сек', str( f.step3 * f.delta_t * f.tx)  + r'сек' ] )
#plt.show()
plt.savefig( 'san_out.png', dpi = 500 )

print( 'time work='+str( hours ) + 'h ' + str( mins ) + 'm ' + str( round(secs, 1) )  + 'sec' )



start = default_timer()

fu.start_values()

M1_out, M2_out, M3_out = fu.cycle_by_t( fu.u, fu.rho, fu.E, fu.P )

stop = default_timer()
total_time = stop - start

# output running time in a nice format.
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

#out from Velocity
###############################

plt.figure( figsize = ( 18, 10 ) )
plt.grid()
plt.xlabel( ' R ' )
plt.ylabel( ' U ' )

number_graph = 0

for i_out in M1_out:

    number_graph += 1

    plt.plot( np.arange( fu.N ) * fu.delta_x, i_out[:-1] )

plt.legend( [ str( fu.step1 * fu.delta_t * fu.tx )  + r'сек', str( fu.step2 * fu.delta_t * fu.tx )  + r'сек', str( fu.step3 * fu.delta_t * fu.tx)  + r'сек' ] )
plt.savefig( 'ichi_ugolok_out.png', dpi = 500 )

plt.figure( figsize = ( 18, 10 ) )
plt.grid()
plt.xlabel( ' R ' )
plt.ylabel( ' U ' )

for i in np.arange( 0, 3, 1 ):

    number_graph += 1

    massive = []

    for j in i_out[:-1]:

        if j != 0:

            massive.append( j + 0.0074 * i )

        else:

            massive.append( 0.0 )

    plt.plot( np.arange( fu.N ) * fu.delta_x, massive )

plt.legend( [ str( fu.step3 * fu.delta_t * fu.tx)  + r'сек, dt= ' + str( fu.delta_t * 1 ), str( fu.step3 * fu.delta_t * fu.tx)  + r'сек, dt= ' + str( fu.delta_t * 2 ),
              str( fu.step3 * fu.delta_t * fu.tx)  + r'сек, dt= ' + str( fu.delta_t * 3 ) ] )
plt.savefig( 'ugolok_out_shodimost_dt.png', dpi = 500 )

plt.figure( figsize = ( 18, 10 ) )
plt.grid()
plt.xlabel( ' R ' )
plt.ylabel( ' U ' )

for i in np.arange( 0, 4, 1 ):

    number_graph += 1

    massive = []

    for j in i_out[:-1]:

        if j != 0:

            massive.append(j + 0.005 * i)

        else:

            massive.append(0.0)

    plt.plot(np.arange(fu.N) * fu.delta_x, massive)

plt.legend( [ str( fu.step3 * fu.delta_t * fu.tx)  + r'сек, dx= ' + str( fu.delta_x * 1 ), str( fu.step3 * fu.delta_t * fu.tx)  + r'сек, dx= ' + str( fu.delta_x * 2 ),
              str( fu.step3 * fu.delta_t * fu.tx)  + r'сек, dx= ' + str( fu.delta_x * 3 ) ] )
plt.savefig( 'ugolok_out_shodimost_dx.png', dpi = 500 )

#out from Density
###################################

plt.figure( figsize = ( 18, 10 ) )
plt.grid()
plt.xlabel( ' R ' )
plt.ylabel( r'$ \rho $' )
plt.yscale( 'log' )

number_graph = 0

for i_out in M2_out:

    number_graph += 1

    plt.plot( np.arange( fu.N - 30 ) * fu.delta_x, np.round( i_out[30:-1], 9) )

plt.axis( [ 0, 31, 1.0e-3, 1.0e-1 ] )
plt.legend( [ str( fu.step1 * fu.delta_t * fu.tx )  + r'сек', str( fu.step2 * fu.delta_t * fu.tx )  + r'сек', str( fu.step3 * fu.delta_t * fu.tx)  + r'сек' ] )
#plt.show()
plt.savefig( 'ni_ugolok_out.png', dpi = 500 )

#out from Energy
#############################

plt.figure( figsize = ( 18, 10 ) )
plt.grid()
plt.xlabel( ' R ' )
plt.ylabel( 'E' )
plt.yscale( 'log' )

number_graph = 0

for i_out in M3_out:

    number_graph += 1

    plt.plot( np.arange( fu.N - 30 ) * fu.delta_x, np.round( i_out[30:-1], 9) )

plt.axis( [ 0, 31, 1.0e-3, 1.0e-1 ] )
plt.legend( [ str( fu.step1 * fu.delta_t * fu.tx )  + r'сек', str( fu.step2 * fu.delta_t * fu.tx )  + r'сек', str( fu.step3 * fu.delta_t * fu.tx)  + r'сек' ] )
#plt.show()
plt.savefig( 'san_ugolok_out.png', dpi = 500 )

print( 'time work='+str( hours ) + 'h ' + str( mins ) + 'm ' + str( round(secs, 1) )  + 'sec' )
