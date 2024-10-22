# ebl.py - Code to calculate tau(E,z) for various photon density models
# Stephen Fegan - sfegan@llr.in2p3.fr - June 2012
# $Id: $

# Updated 2014-10-22 : Python3.4

import math
import scipy
import scipy.integrate
import scipy.interpolate
import numpy.polynomial.chebyshev
import bisect
import sys
#import scipy.optimize
#import scipy.stats

# Units: [E] Energy in      eV
#        [D] Distance in    cm

m_e      = 510998.909642645     # eV       - Mass of electron
sigma_t  = 6.652458554889e-25   # cm^2     - Thompson cross-section
c        = 2.99792458e+10       # cm s^-1  - Speed of light
ly       = c*3600*24*365.25     # cm       - Lightyear
pc       = 3.26156377694477*ly  # cm       - Parsec
km_s_Mpc = 0.1/pc               # s^-1     - Conventional units of Hubble const
k_B      = 8.61734229648141e-05 # eV K^-1  - Boltzmann constant
h        = 4.13566733363251e-15 # eV s     - Planck constant
hc       = h*c                  # eV cm    - Planck times c
erg      = 6.2415093E+11        # eV       - 1 erg in eV

tol_int_mu      = 1e-2
tol_int_epsilon = 1e-2
tol_int_zprime  = 1e-3
tol_int_dldz    = 1e-3

def extrap1dQ(interpolator, al=None, ah=None):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            slope1 = (ys[1]-ys[0])/(xs[1]-xs[0])
            slope2 = (ys[2]-ys[1])/(xs[2]-xs[1])
            if al==None:
                a = (slope2-slope1)/((xs[2]-xs[0])/2.0)
            else:
                a = al
            b = slope1 - 2.0*a*xs[0]
            c = ys[0] - (a*xs[0] + b)*xs[0]
            return (a*x+b)*x+c
        elif x > xs[-1]:
            slope1 = (ys[-2]-ys[-1])/(xs[-2]-xs[-1])
            slope2 = (ys[-3]-ys[-2])/(xs[-3]-xs[-2])
            if ah==None:
                a = (slope2-slope1)/((xs[-3]-xs[-1])/2.0)
            else:
                a = ah
            b = slope1 - 2.0*a*xs[-1]
            c = ys[-1] - (a*xs[-1] + b)*xs[-1]
            return (a*x+b)*x+c
        else:
            return interpolator(x)

    def ufunclike(x):
        return pointwise(x)

    return ufunclike

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(x):
        return pointwise(x)

    return ufunclike

def n_epsilon_z_bb(epsilon, z, T0=2.725):
    norm = 8*math.pi/hc**3
    T = T0*(1.0+z)
    x = epsilon/(k_B*T);
    try:
        n = norm*epsilon**2/(math.exp(x) - 1.0)
    except OverflowError:
        return 0
    return n

def n_epsilon_logeMdnde_functor_no_z(epsilon, z, logeMdnde, M=1):
    #print(epsilon, math.log10(epsilon), logeMdnde(math.log10(epsilon)))
    logepsilon = math.log10(epsilon)
    return math.pow(10.0,logeMdnde(logepsilon) - M*logepsilon)

def n_epsilon_logeMdnde_functor_interp_z(epsilon, z, allz, logeMdnde, M=1,
                                         z_power=2, e_power=1):
    i = bisect.bisect_left(allz, z)-1
    z0 = allz[i]
    z1 = allz[i+1]
    x = (z-z0)/(z1-z0)
    logopz = math.log10(1.0+z)
    logopz0 = math.log10(1.0+z0)-logopz
    logopz1 = math.log10(1.0+z1)-logopz
    logepsilon = math.log10(epsilon)
    logepsilon0 = logepsilon + e_power*logopz0
    logepsilon1 = logepsilon + e_power*logopz1
    y0 = logeMdnde[i](logepsilon0)   - M*logepsilon0 - z_power*logopz0
    y1 = logeMdnde[i+1](logepsilon1) - M*logepsilon1 - z_power*logopz1
    return math.pow(10.0,(1.0-x)*y0+x*y1)

def n_epsilon_logeMdnde_quadratic_no_z(epsilon, z,
                                       loge_peak, logeMdnde_peak, a, M=1):
    x = math.log10(epsilon)
    return math.pow(10.0,a*(x-loge_peak)**2+logeMdnde_peak)/epsilon**M

def n_epsilon_sum(epsilon, z, n_epsilons):
    n = 0
    for n_epsilon in n_epsilons:
        n += n_epsilon(epsilon, z)
    return n

def n_epsilon_zevolution(epsilon, z, n_epsilon_base, z_power=2, e_power=1):
    opz = 1.0+z
    return n_epsilon_base(epsilon/math.pow(opz,e_power),0)*math.pow(opz,z_power)

def sigma_gg(s0,mu):
    b = math.sqrt(1.0 - 2.0/(s0*(1.0-mu)))
    return 3.0*sigma_t/16.0*(1.0-b**2)*\
        (2.0*b*(b**2-2.0)+(3.0-b**4)*math.log((1.0+b)/(1.0-b)))

def integrand_mu(mu,s0):
    b = math.sqrt(1.0 - 2.0/(s0*(1.0-mu)))
    return (1.0-mu)/2.0 *3.0*sigma_t/16.0 * (1.0-b**2)*\
        (2.0*b*(b**2-2.0)+(3.0-b**4)*math.log((1.0+b)/(1.0-b)))

def integral_mu(s0,tol=tol_int_mu):
    mu_th = 1.0 - 2.0/s0
    #print("Mu_th:", mu_th)
    ival, ierr = \
    scipy.integrate.quad(integrand_mu, -1.0, mu_th,
                         args = s0, epsrel=tol, epsabs=0)
    #print("Mu integral:",ival, ierr)
    return ival

def integrand_phi0(s):
    b = math.sqrt(1.0 - 1.0/s)
    return 2.0*b*(b**2-2.0)+(3.0-b**4)*math.log((1.0+b)/(1.0-b))

def integral_phi0(s0,tol=tol_int_mu):
    ival, ierr = \
        scipy.integrate.quad(integrand_phi0, 1, s0, epsrel=tol, epsabs=0)
    return ival

def series_sum_phi0(s0,tol=tol_int_mu):
    # Formula from Gould-Schreder, corrected as per Dermer's book pg 230
    b0sq = 1.0 - 1.0/s0
    b0 = math.sqrt(b0sq)
    w0 = (1.0 + b0)/(1.0 - b0)
    logw0 = math.log(w0)
    phi0 = (2.0*s0 - 2.0 + 1.0/s0)*logw0 \
        + 2.0*(1.0-2.0*s0)*b0 \
        + logw0*(4.0*math.log(1.0+w0) - 3.0*logw0) \
        - math.pi*math.pi/3.0;

    x = 1.0/w0
    n = 1
    xn = x
    while n<10000:
        corr = 4.0/(n*n)*xn
        if n&1: phi0 += corr
        else: phi0 -= corr
        #print(phi0, corr, corr/phi0)
        if math.fabs(corr/phi0)<tol:
            return phi0, n
        xn *= x
        n+=1
    raise ValueError("Limit on series iterations exceeded")

def series_sum_phi0_2(s0,tol=tol_int_mu):
    # Formula from Gould-Schreder, corrected as per Dermer's book pg 230
    # Modified by applying transformations from Abromowitz & Stegun
    b0sq = 1.0 - 1.0/s0
    b0 = math.sqrt(b0sq)
    w0 = (1.0 + b0)/(1.0 - b0)
    logw0 = math.log(w0)
    opw0 = 1.0+w0
    logopw0 = math.log(opw0)
    phi0 = (2.0*s0 - 2.0 + 1.0/s0)*logw0 \
        + 2.0*(1.0-2.0*s0)*b0 \
        - 1.0*logw0*logw0 + 2.0*logopw0*logopw0 \
        - math.pi*math.pi/3.0;

    x = 1.0/opw0
    n = 1
    xn = x
    while n<10000:
        corr = 4.0/(n*n)*xn
        phi0 += corr
        #print(phi0, corr, corr/phi0)
        if math.fabs(corr/phi0)<tol:
            return phi0, n
        xn *= x
        n+=1
    raise ValueError("Limit on series iterations exceeded")        

def integrand_epsilon(epsilon,Eg,n_epsilon):
    s0 = Eg*epsilon/(m_e*m_e)
    if s0>1.0:
        C = 3.0*sigma_t/(8.0*s0*s0)
#        I1 = integral_mu(s0)
#        I2,n2 = series_sum_phi0(s0)
#        I2 *= C
        I3,n3 = series_sum_phi0_2(s0)
        I3 *= C
#        print(epsilon,I1,I2,I3,n2,n3)
        return n_epsilon(epsilon) * I3
    return 0;

def integral_epsilon(Eg,n_epsilon,tol=tol_int_epsilon):
    #print("")
    #print("")
    epsilon_th = m_e**2/Eg
    ival, ierr = \
        scipy.integrate.quad(integrand_epsilon, epsilon_th, scipy.inf,
                             args = (Eg, n_epsilon), epsrel=tol, epsabs=0)
    #print("Epsilon integral: ",ival, ierr)
    return ival

def mean_epsilon(Eg,n_epsilon,tol=tol_int_epsilon):
    #print("")
    #print("")
    epsilon_th = m_e**2/Eg
    i_epsilon, ierr = \
        scipy.integrate.quad(lambda epsilon: epsilon*integrand_epsilon(epsilon,Eg,n_epsilon), epsilon_th, scipy.inf,
                             epsrel=tol, epsabs=0)
    i_norm, ierr = \
        scipy.integrate.quad(integrand_epsilon, epsilon_th, scipy.inf,
                             args = (Eg, n_epsilon), epsrel=tol, epsabs=0)
    return i_epsilon/i_norm

def dldz_frw(z,omega_m=0.25,omega_r=0.0,omega_l=0.75,H0=74.2):
    H0 *= km_s_Mpc
    opz = 1.0+z
    return c/(H0*opz*math.sqrt(opz**2*(omega_m*z+1.0) + \
                                   z*(2.0+z)*(opz**2*omega_r - omega_l)))

def integral_dldz_dz(z, dldz, tol=tol_int_dldz):
    ival, ierr = scipy.integrate.quad(dldz, 0, z, epsrel=tol)
    return ival

def distance_comoving_radial(z, dldz, tol=tol_int_dldz):
    ival, ierr = scipy.integrate.quad(lambda z: (1+z)*dldz(z), 0, z, epsrel=tol)
    return ival

def distance_comoving_transverse_flat(z, dldz, tol=tol_int_dldz):
    return distance_comoving_radial(z, dldz, tol)

def distance_luminosity_flat(z, dldz, tol=tol_int_dldz):
    return (1+z)*distance_comoving_transverse_flat(z, dldz, tol)

def distance_angular_flat(z, dldz, tol=tol_int_dldz):
    return distance_comoving_transvers_flat(z, dldz, tol)/(1+z)**2

def integrand_zprime(zprime,Eg,n_epsilon_zprime,dldz):
    n_epsilon = lambda epsilon: n_epsilon_zprime(epsilon,zprime);
    return dldz(zprime)*integral_epsilon(Eg*(1.0+zprime),n_epsilon)

def integral_zprime(z,Eg,n_epsilon_zprime,dldz=dldz_frw,tol=tol_int_zprime):
    ival, ierr = \
        scipy.integrate.quad(integrand_zprime, 0, z,
                             args=(Eg, n_epsilon_zprime,dldz),
                             epsrel=tol, epsabs=0)
    #print("Zprime integral: ",ival, ierr)
    return ival

def tau(z,Eg,n_epsilon_zprime,dldz=dldz_frw):
    return integral_zprime(z,Eg,n_epsilon_zprime,dldz)

def dtau_dlogEg(z,Eg,n_epsilon_zprime,dldz=dldz_frw):
    dx = 1e-3 
    x = math.log(Eg)
    x0 = x - dx/2.0
    x1 = x + dx/2.0
    t0 = integral_zprime(z,math.exp(x0),n_epsilon_zprime,dldz_frw)
    t1 = integral_zprime(z,math.exp(x1),n_epsilon_zprime,dldz_frw)
    return (t1-t0)/(x1-x0)

def d2tau_dlogEg_dz(z,Eg,n_epsilon_zprime,dldz=dldz_frw):
    dx = 1e-3 
    x = math.log(Eg)
    x0 = x - dx/2.0
    x1 = x + dx/2.0
    dtdz0 = integrand_zprime(z,math.exp(x0),n_epsilon_zprime,dldz_frw)
    dtdz1 = integrand_zprime(z,math.exp(x1),n_epsilon_zprime,dldz_frw)
    return (dtdz1-dtdz0)/(x1-x0)

def read_francescini_density(filename):
    file = open(filename,'r');
    got_header = False
    z = []
    x = []
    y = []
    for line in file:
        line = line.split('%',1)[0];
        line = line.split('#',1)[0];
        line.lstrip()
        bits = line.split();
        if len(bits)==0: continue
        if not got_header:
            for i in range(len(bits)):
                if i%2==1:
                    z.append(float(bits[i]))
                    x.append([])
                    y.append([])
            got_header = True
        else:
            for i in range(len(bits)):
                j = i//2
                if i%2==0: x[j].append(float(bits[i]))
                else: y[j].append(float(bits[i]))
    ednde = []
    for (xi, yi) in zip(x,y):
        ednde.append(extrap1dQ(scipy.interpolate.interp1d(xi,yi),-5.0,-5.0))
    return z, ednde

if __name__ == "__main__":
    #print(integral_l(3,lambda z: dldz_frw(z,0.27,0,0.73,71))/ly/1e9)
    #print(integral_l(0.3, dldz_frw)/ly/1e9)

    allz, logednde = read_francescini_density('francescini_ednde.dat')

    ne_f = lambda e,z: \
        n_epsilon_logeMdnde_functor_interp_z(e, z, allz, logednde) #, e_power=0, z_power=0)
    ne_Qo = lambda e,z: \
        n_epsilon_logeMdnde_quadratic_no_z(e, z, -0.05391, -2.508, -1.965, 2)
    ne_Qi = lambda e,z: \
        n_epsilon_logeMdnde_quadratic_no_z(e, z, -2.103, -2.231, -2.828, 2)
    ne_CMB = n_epsilon_z_bb

    ne1 = lambda e,z: n_epsilon_sum(e, z, [ne_f, ne_CMB])
    ne2 = lambda e,z: n_epsilon_sum(e, z, [ne_Qo, ne_Qi, ne_CMB])

    if False:
        x=-6
        while x<2:
            e=math.pow(10.0,x)
            print(x, e, ne1(e,0), ne2(e,0))
            x += 0.02
        exit(0)

    if False:
        neA = lambda e,z: n_epsilon_logeMdnde_functor_no_z(e, z, logednde[0])
        neB = lambda e,z: n_epsilon_logeMdnde_functor_no_z(e, z, logednde[2])
        neC = lambda e,z: n_epsilon_zevolution(e, z, neA, z_power=2.0)
        neD = lambda e,z: n_epsilon_zevolution(e, z, neA, z_power=2.0-1.2)
        z0=0.4
        x=-2.8
        while x<1.0:
            e=math.pow(10.0,x)
            print(x, e, neA(e,z0), neB(e,z0), neC(e,z0), neD(e,z0))
            x += 0.02
        exit(0)

    if False:
        s0=1.001;
        while s0<1000000:
            b0sq = 1.0 - 1.0/s0
            b0 = math.sqrt(b0sq)
            w0 = (1.0 + b0)/(1.0 - b0)
            I1, n1 = series_sum_phi0_2(s0,tol=1e-2)
            I2, n2 = series_sum_phi0(s0,tol=1e-2)
            I3 = integral_mu(s0,tol=1e-12)
            I4 = integral_phi0(s0,tol=1e-12)
            print(s0, w0, I3, I4, I1, n1, I2, n2)
            s0 *= 1.01
        exit(0)

    if False:
        sinv=0.00005;
        x = []
        y = []
        while sinv<1:
            I = integral_phi0(1/sinv,tol=1e-12)
            x.append(sinv)
            y.append(I*sinv*sinv);
            sinv += 0.0001
        coeff, info = numpy.polynomial.chebyshev.chebfit(x, y, 1024, full=True)
        for xi, yi in zip(x,y):
            yc = numpy.polynomial.chebyshev.chebval(xi, coeff)
            print(1.0/xi, yi, yc)
        exit(0)

    if False:
        s0 = 10.0
        mu_th = 1.0 - 2.0/s0
        mu = -1
        while mu < mu_th:            
            I1, ierr = \
                scipy.integrate.quad(integrand_mu, mu, mu_th,
                                     args = s0, epsrel=1e-8, epsabs=0)
            s = s0*(1.0-mu)/2.0
            I2 = integral_phi0(s, 1e-8)
            I3, n3 = series_sum_phi0(s, 1e-8)
            print(mu, I1, I2, I3)
            mu += 0.01
        exit(0)

    if False:
        try:
            zsrc = 0.1
            if len(sys.argv) > 1:
                zsrc = float(sys.argv[0])
            
            x = 10.0
            while x<14.0:
                Eg = 2.0*math.pow(10.0,x)
            
                if True:
                    print('%8.3g %g %g'%(Eg/1e12, \
                                         integral_zprime(zsrc, Eg, ne1), \
                                         integral_zprime(zsrc, Eg, ne2)))
            
                if False:
                    print('%8.3g %g %g %g'%(Eg/1e12, \
                                            d2tau_dlogEg_dz(0, Eg, ne1), \
                                            d2tau_dlogEg_dz(0, Eg, ne2), \
                                            d2tau_dlogEg_dz(0, Eg, ne_Qo)))
            
                x+=.08
        except KeyboardInterrupt:
            pass

    if True:
        try:
            zsrc = 0.1
            etev = 0.3
            if len(sys.argv) > 1:
                zsrc = float(sys.argv[1])
            if len(sys.argv) > 2:
                etev = float(sys.argv[2])
            print('%g %g %g'%(zsrc, etev, dtau_dlogEg(zsrc, etev*1e12, ne1)))
        except KeyboardInterrupt:
            pass
