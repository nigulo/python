import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import scipy.special as special
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import cm

'''
 Return a binomial coefficient
'''
def binomial_coef(n, k):

    if k > n:
        return 0.0
    kk = k
        
    if k > n/2:
        kk = n-k
            
    binomial_coef = 1.0
    for i in np.arange(1, kk + 1):
        binomial_coef *= (n-kk+i) / i

    return binomial_coef

'''
 Return if a number is odd
'''
def odd(m):
    return (m % 2) != 0


'''   
 Return the Vnm(r,f) function of the theory
'''
def Vnmf(radio, f, n, m):
    #real(kind=8) :: radio(:,:), f
    #integer :: n, m, l, j, lmax, p, q, ix, iy
    #real(kind=8) :: t1, t2, t3, t4, ulj, sum(size(radio(:,1)),size(radio(1,:))), epsm
    #complex(kind=8) :: ii, Vnm(size(radio(:,1)),size(radio(1,:)))

    p = int(0.5*(n-abs(m)))
    q = int(0.5*(n+abs(m)))
    
    epsm = 1.0
    if m < 0.0 and odd(m):
        epsm = -1.0
    
    lmax = 3.5*abs(f)+1
    
    ii = 1.j
    
    Vnm = np.zeros_like(radio, dtype='complex')
    
    for l in np.arange(1, lmax + 1):

        sum_ = np.zeros_like(radio)
        for j in np.arange(0, p + 1):
            if p-j < l:
                t1 = binomial_coef(abs(m)+j+l-1,l-1)
                t2 = binomial_coef(j+l-1,l-1)
                t3 = binomial_coef(l-1,p-j)
                t4 = binomial_coef(q+l+j,l)
                ulj = (-1.0)**p * (abs(m)+l+2*j) * t1 * t2 * t3 / t4
                for ix in np.arange(0, np.shape(radio)[0]):#size(radio(:,1)))
                    for iy in np.arange(0, np.shape(radio)[1]):#size(radio(1,:))
                        #sum_[ix,iy] += ulj * special.iv(abs(m)+l+2*j, 2.0*np.pi*(radio[ix,iy]+1.e-10)) / (l*(2.0*np.pi*(radio[ix,iy]+1.e-10))**l)
                        sum_[ix,iy] += ulj * special.iv(abs(m)+l+2*j, 2.0*np.pi*(radio[ix,iy])) / (l*(2.0*np.pi*(radio[ix,iy]))**l)
        Vnm += sum_ * (-2.0*ii*f)**(l-1)
        
    Vnm *= epsm * np.exp(ii*f)
    return Vnm

'''
 Return the n index associated to a Noll index j
'''
def noll_n(j):
    return int(np.floor(np.sqrt(2.0*j-0.75)-0.5))
        

'''
 Return the m index associated to a Noll index j
'''
def noll_m(j):
    n = noll_n(j)
    nn = (n-1)*(n+2)/2
    noll_m = j-nn-2
    if np.ceil(n/2.) != np.floor(n/2.):
        noll_m = 2*int(noll_m/2.)+1
    else:
        noll_m = 2*int((noll_m+1)/2.)

    if np.ceil(j/2.) != np.floor(j/2.):
        noll_m = -noll_m

    return noll_m        

def reverse_colourmap(cmap, name = 'my_cmap_r'):
     return mpl.colors.LinearSegmentedColormap(name, cm.revcmap(cmap._segmentdata))

my_cmap = reverse_colourmap(plt.get_cmap('binary'))#plt.get_cmap('winter')

zoom_factor = 1.0
jmax = 5
arcsec_per_px = 0.055
diameter = 1.0
wavelength = 5250.0
nx = 100
F_D = 1.0
# Ask for some data
if False:
    jmax = int(input('Maximum Noll index? '))
    arcsec_per_px = float(input('Arcsec per pixel? '))
    diameter = float(input('Telescope diameter [m]? '))
    wavelength = float(input('Wavelength [A]? '))
    nx = int(input('Number of pixel of images? '))
    F_D = float(input('F/D? '))


diffraction = 1.22 * wavelength * 1e-8 / diameter
diffraction = 206265.0 * diffraction

print('Diffraction limit ["]=', diffraction)

# Generate pupil plane
x_diff = np.zeros(nx)
y_diff = np.zeros(nx)
for i in np.arange(0, nx):
    x_diff[i] = arcsec_per_px*i
    y_diff[i] = arcsec_per_px*i

x_diff = x_diff - x_diff[int(nx/2)]
y_diff = y_diff - y_diff[int(nx/2)]

x_diff = x_diff / diffraction
y_diff = y_diff / diffraction

radio = np.zeros((nx,nx))
phi = np.zeros((nx,nx))
#V_n_m = np.zeros((nx,nx), dtype='complex'))
#V_np_mp = np.zeros((nx,nx), dtype='complex'))
#c_sum = np.zeros((nx,nx))
#s_sum = np.zeros((nx,nx))
#xi = np.zeros((nx,nx))
#psi = np.zeros((nx,nx))
#X = np.zeros((nx,nx))
#Y = np.zeros((nx,nx))
in_fft = np.zeros((nx,nx), dtype='complex')
out_fft = np.zeros((nx,nx), dtype='complex')

for i in np.arange(0, nx):
    for j in np.arange(0, nx):
        radio[i,j] = np.sqrt(x_diff[i]**2 + y_diff[j]**2)
        phi[i,j] = np.arctan2(y_diff[j], x_diff[i])

radio = radio * 3.8317 / (2.0 * np.pi)

# Generate the two focus+defocused PSFs
for loop_defocus in ["focus", "defocus"]:
    f = 0.0
    defocus_mm = 0.0
    
    if loop_defocus == "defocus":
        d_lambda = 8.0 * F_D**2
        f = np.pi * d_lambda / (4 * (F_D)**2)
        defocus_mm = d_lambda * wavelength*1.e-7

        print('Defocus in mm = ', d_lambda * wavelength*1.e-7)
        print('Defocus f = ', f)

#    write(s_jmax,FMT='(I2)') jmax
#    write(s_arcsec_per_px,FMT='(F6.4)') arcsec_per_px
#    write(s_nx,FMT='(I3)') nx
#    write(s_diameter,FMT='(I3)') diameter
#    write(s_F_D,FMT='(F4.1)') F_D
#    write(s_wavelength,FMT='(F6.1)') wavelength
#    write(s_defocus,FMT='(F6.3)') defocus_mm

#    open(unit=12,file='PSF_BASIS/psf_basis_jmax_'//trim(adjustl(s_jmax))//'_arcsec_per_px_'//trim(adjustl(s_arcsec_per_px))//'_size_'//trim(adjustl(s_nx))//'_D_'//trim(adjustl(s_diameter))//'_FD_'//trim(adjustl(s_F_D))//'_lambda_'//trim(adjustl(s_wavelength))//'_defocus_'//trim(adjustl(s_defocus))//'.bin',&
#        action='write',form='unformatted')

#    write(12) nx, nx, 2, jmax, jmax

    # Generate all the basis functions
    
    ncols = jmax
    nrows = int(jmax*jmax / ncols)
    if ncols * nrows < jmax:
        ncols += 1
        
    assert(ncols * nrows >= jmax * jmax)
    fig_x, axes_x = plt.subplots(nrows=nrows, ncols=ncols)
    fig_x.set_size_inches(ncols*3, nrows*3)
    
    fig_y, axes_y = plt.subplots(nrows=nrows, ncols=ncols)
    fig_y.set_size_inches(ncols*3, nrows*3)
    
    extent=[0., 1., 0., 1.]
    plot_aspect=(extent[1]-extent[0])/(extent[3]-extent[2])#*2/3 

    row = 0
    col = 0
    for j in np.arange(1, jmax + 1):
        m = noll_m(j)
        n = noll_n(j)

        V_n_m =  Vnmf(radio, f, n, m)
        
        for k in np.arange(1, jmax + 1):
            m_p = noll_m(k)
            n_p = noll_n(k)
            print(n, m, n_p, m_p)


            V_np_mp = Vnmf(radio, f, n_p, m_p)

            ca = np.cos(0.5*(m+m_p)*np.pi)
            sa = np.sin(0.5*(m+m_p)*np.pi)
            if abs(ca) < 1.0e-14:
                ca = 0.0
            if abs(sa) < 1.0e-14:
                sa = 0.0

            c_sum = ca*np.cos((m-m_p)*phi) - sa*np.sin((m-m_p)*phi)
            s_sum = sa*np.cos((m-m_p)*phi) + ca*np.sin((m-m_p)*phi)

            xi = V_n_m.imag * V_np_mp.imag + V_n_m.real * V_np_mp.real
            psi = V_n_m.real * V_np_mp.imag - V_n_m.imag * V_np_mp.real
        
            X = 8.0 * (-1.0)**m_p * (c_sum * xi + s_sum * psi)
            Y = 8.0 * (-1.0)**m_p * (c_sum * psi - s_sum * xi)
            
            # Do the plotting #################################################

            ax_x = axes_x[row][col]
            ax_y = axes_y[row][col]
            col += 1
            if col >= ncols:
                col = 0
                row += 1
            
            title_x = r'$X^{'+ str(m) + r',' + str(m_p) + r'}_{' + str(n) + ',' + str(n_p) + r'}$'
            title_y = r'$Y^{'+ str(m) + r',' + str(m_p) + r'}_{' + str(n) + ',' + str(n_p) + r'}$'
            ax_x.text(0.95, 0.01, title_x,
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax_x.transAxes,
                    color='yellow', fontsize=10)
    
            ax_y.text(0.95, 0.01, title_y,
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax_y.transAxes,
                    color='yellow', fontsize=10)

            center = nx/2
            zoom_radius = nx*zoom_factor/2
            left = int(center-zoom_radius)
            right = int(center+zoom_radius)
            ax_x.imshow(X[left:right,left:right].T,extent=extent,cmap=my_cmap,origin='lower')
            ax_x.set_aspect(aspect=plot_aspect)

            ax_y.imshow(Y[left:right,left:right].T,extent=extent,cmap=my_cmap,origin='lower')
            ax_y.set_aspect(aspect=plot_aspect)

            ###################################################################

            # Do the FFT and save the results
            
            #FX = fft.fft2(X)
            #FX = np.roll(np.roll(FX, int(nx/2), axis=0), int(nx/2), axis=1)

            #FY = fft.fft2(Y)
            #FY = np.roll(np.roll(FY, int(nx/2), axis=0), int(nx/2), axis=1)

    fig_x.savefig("psf_"+loop_defocus+"_x.png")
    plt.close(fig_x)

    fig_y.savefig("psf_"+loop_defocus+"_y.png")
    plt.close(fig_y)
