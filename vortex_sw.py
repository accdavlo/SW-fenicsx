import numpy as np
gravity = 9.81
pi= np.pi

def lambda_vortex2(r):
    lam = (20.*np.cos(r))/3. + (27.*np.cos(r)**2.)/16. + (4.*np.cos(r)**3)/9.+ np.cos(r)**4/16. + (20.*r*np.sin(r))/3. \
            + (35.*r**2)/16. + (27.*r*np.cos(r)*np.sin(r))/8. + (4.*r*np.cos(r)**2*np.sin(r))/3. + (r*np.cos(r)**3*np.sin(r))/4.
    return lam

def analytic_travelling_vortexH(x, y, x0,y0, t = 0.):
    # fonction parameter
    Xc = [x0,y0]
    r0 = 0.45
    deltah = 0.1
    w = pi/r0         # angular wave frequency
    Gam=(12.*pi*np.sqrt(deltah*gravity))/(np.sqrt(315.*pi**2. - 2048.))/r0   # vortex intensity parameter
    hc = 1.          # stady state 
    uc = 0.          # initial angular speed
    vc = 0.          # initial angular speed
    # coordinates centered with the vortex core
    xx = x - Xc[0] - uc*t
    yy = y - Xc[1] - vc*t
    # distance from the vortex core
    r = np.sqrt(xx*xx + yy*yy)
    H = hc
    H += (r < r0)*(1./gravity * (Gam/w)**2. * \
        (lambda_vortex2(w*r) - lambda_vortex2(pi)))
    return H
def analytic_travelling_vortexU(x, y, x0,y0, t = 0.):
    # fonction parameter
    Xc = [x0,y0]
    r0 = 0.45
    deltah = 0.1
    w = pi/r0         # angular wave frequency
    Gam=(12.*pi*np.sqrt(deltah*gravity))/(np.sqrt(315.*pi**2. - 2048.))/r0   # vortex intensity parameter
    hc = 1.          # stady state 
    uc = 0.  # initial angular speed
    vc = 0.          # initial angular speed
    # coordinates centered with the vortex core
    xx = x - Xc[0] - uc*t
    yy = y - Xc[1] - vc*t
    # distance from the vortex core
    r = np.sqrt(xx*xx + yy*yy)
    H = hc
    up = 0.
    H += (r < r0)*(1./gravity * (Gam/w)**2. * \
        (lambda_vortex2(w*r) - lambda_vortex2(pi)))
    up += (r < r0)*(Gam*(1. + np.cos(w*r))**2*(-yy))
    U = uc + up
    return U
def analytic_travelling_vortexV( x, y, x0,y0, t = 0.):
    # fonction parameter
    Xc = [x0,y0]
    r0 = 0.45
    deltah = 0.1
    w = pi/r0         # angular wave frequency
    Gam=(12.*pi*np.sqrt(deltah*gravity))/(np.sqrt(315.*pi**2. - 2048.))/r0   # vortex intensity parameter
    hc = 1.          # stady state 
    uc = 0.  # initial angular speed
    vc = 0.          # initial angular speed
    # coordinates centered with the vortex core
    xx = x - Xc[0] - uc*t
    yy = y - Xc[1] - vc*t
    # distance from the vortex core
    r = np.sqrt(xx*xx + yy*yy)
    H = hc
    vp = 0.
        
    H += (r < r0)*(1./gravity * (Gam/w)**2. * \
        (lambda_vortex2(w*r) - lambda_vortex2(pi)))
    vp += (r < r0)*(Gam*(1. + np.cos(w*r))**2*(xx))
    V = vc + vp
    return V   