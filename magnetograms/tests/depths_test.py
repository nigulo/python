import sys
sys.path.append('..')
import numpy as np
import unittest
import sys
sys.path.append('../../utils')
import depths


class test_depths(unittest.TestCase):

    
    def test_calc_b_d_az(self):
        
        n1 = 10
        n2 = 10
        
        x1_range = 1.0
        x2_range = 1.0

        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x_mesh = np.meshgrid(x2, x1)[::-1]
        x_grid = np.dstack(x_mesh)
        
        bx = np.random.normal(size=(n1, n2))
        by = np.random.normal(size=(n1, n2))
        bz = np.random.normal(size=(n1, n2))
        
        dpths = depths.depths(x_grid, bx, by, bz, prior_prec=1.)
        dpths.i = np.random.randint(1, n1-2)
        dpths.j = np.random.randint(1, n2-2)

        params = np.random.normal(size=16)
        b_derivs = params[:8]
        dz = params[8:16]
        b, d, az = dpths.calc_b_d_az(b_derivs, dz)

        i = dpths.i
        j = dpths.j
        
        b_expected = np.zeros(24)
        b_expected[0] = bx[i, j] - bx[i-1, j]
        b_expected[1] = bx[i+1, j] - bx[i, j]

        b_expected[2] = bx[i, j] - bx[i, j-1]
        b_expected[3] = bx[i, j+1] - bx[i, j]
        
        b_expected[4] = by[i, j] - by[i-1, j]
        b_expected[5] = by[i+1, j] - by[i, j]

        b_expected[6] = by[i, j] - by[i, j-1]
        b_expected[7] = by[i, j+1] - by[i, j]

        b_expected[8] = bz[i, j] - bz[i-1, j]
        b_expected[9] = bz[i+1, j] - bz[i, j]

        b_expected[10] = bz[i, j] - bz[i, j-1]
        b_expected[11] = bz[i, j+1] - bz[i, j]


        b_expected[12] = bx[i, j] - bx[i-1, j-1]
        b_expected[13] = bx[i+1, j+1] - bx[i, j]

        b_expected[14] = bx[i, j] - bx[i+1, j-1]
        b_expected[15] = bx[i-1, j+1] - bx[i, j]

        b_expected[16] = by[i, j] - by[i-1, j-1]
        b_expected[17] = by[i+1, j+1] - by[i, j]

        b_expected[18] = by[i, j] - by[i+1, j-1]
        b_expected[19] = by[i-1, j+1] - by[i, j]

        b_expected[20] = bz[i, j] - bz[i-1, j-1]
        b_expected[21] = bz[i+1, j+1] - bz[i, j]

        b_expected[22] = bz[i, j] - bz[i+1, j-1]
        b_expected[23] = bz[i-1, j+1] - bz[i, j]

       
        dx = x_grid[1, 0, 0] - x_grid[0, 0, 0]
        dy = x_grid[0, 1, 1] - x_grid[0, 0, 1]

        d_expected = np.zeros(24)
        d_expected[0] = -b_derivs[0]*dx
        d_expected[1] = b_derivs[0]*dx
        
        d_expected[2] = -b_derivs[1]*dy
        d_expected[3] = b_derivs[1]*dy

        d_expected[4] = -b_derivs[3]*dx
        d_expected[5] = b_derivs[3]*dx

        d_expected[6] = -b_derivs[4]*dy
        d_expected[7] = b_derivs[4]*dy

        d_expected[8] = -b_derivs[6]*dx
        d_expected[9] = b_derivs[6]*dx

        d_expected[10] = -b_derivs[7]*dy
        d_expected[11] = b_derivs[7]*dy
        
        d_expected[12] = -b_derivs[0]*dx - b_derivs[1]*dy
        d_expected[13] = b_derivs[0]*dx + b_derivs[1]*dy

        d_expected[14] = b_derivs[0]*dx - b_derivs[1]*dy
        d_expected[15] = -b_derivs[0]*dx + b_derivs[1]*dy
        
        d_expected[16] = -b_derivs[3]*dx - b_derivs[4]*dy
        d_expected[17] = b_derivs[3]*dx + b_derivs[4]*dy

        d_expected[18] = b_derivs[3]*dx - b_derivs[4]*dy
        d_expected[19] = -b_derivs[3]*dx + b_derivs[4]*dy
        
        d_expected[20] = -b_derivs[6]*dx - b_derivs[7]*dy
        d_expected[21] = b_derivs[6]*dx + b_derivs[7]*dy

        d_expected[22] = b_derivs[6]*dx - b_derivs[7]*dy
        d_expected[23] = -b_derivs[6]*dx + b_derivs[7]*dy

        
        dbx_dz = b_derivs[2]
        dby_dz = b_derivs[5]
        dbz_dz = -b_derivs[0] - b_derivs[4]
        az_expected = np.zeros(24)


        az_expected[0] = dbx_dz*dz[0]
        az_expected[1] = dbx_dz*dz[1]
        az_expected[2] = dbx_dz*dz[2]
        az_expected[3] = dbx_dz*dz[3]

        az_expected[4] = dby_dz*dz[0]
        az_expected[5] = dby_dz*dz[1]
        az_expected[6] = dby_dz*dz[2]
        az_expected[7] = dby_dz*dz[3]

        az_expected[8] = dbz_dz*dz[0]
        az_expected[9] = dbz_dz*dz[1]
        az_expected[10] = dbz_dz*dz[2]
        az_expected[11] = dbz_dz*dz[3]

        az_expected[12] = dbx_dz*dz[4]
        az_expected[13] = dbx_dz*dz[5]
        az_expected[14] = dbx_dz*dz[6]
        az_expected[15] = dbx_dz*dz[7]

        az_expected[16] = dby_dz*dz[4]
        az_expected[17] = dby_dz*dz[5]
        az_expected[18] = dby_dz*dz[6]
        az_expected[19] = dby_dz*dz[7]

        az_expected[20] = dbz_dz*dz[4]
        az_expected[21] = dbz_dz*dz[5]
        az_expected[22] = dbz_dz*dz[6]
        az_expected[23] = dbz_dz*dz[7]


        np.testing.assert_array_almost_equal(b, b_expected)
        np.testing.assert_array_almost_equal(d, d_expected)
        np.testing.assert_array_almost_equal(az, az_expected)
        
    '''
    def test_loss_fn(self):
        
        n1 = 10
        n2 = 10
        
        x1_range = 1.0
        x2_range = 1.0

        x1 = np.linspace(0, x1_range, n1)
        x2 = np.linspace(0, x2_range, n2)
        x_mesh = np.meshgrid(x2, x1)[::-1]
        x_grid = np.dstack(x_mesh)
        
        bx = np.random.normal(size=(n1, n2))
        by = np.random.normal(size=(n1, n2))
        bz = np.random.normal(size=(n1, n2))
        
        d = depths.depths(x_grid, bx, by, bz, prior_prec=1.)
        d.i = np.random.randint(1, n1-1)
        d.j = np.random.randint(1, n2-1)
        d.estimate()

        params = np.random.normal(size=16)
        loss = d.loss_fn(params)
        
        b_derivs = params[:8]
        dz = params[8:16]
        
        i = d.i
        j = d.j
        b = np.zeros(24)
        
        b[0] = bx[i, j] - bx[i-1, j]
        b[1] = bx[i+1, j] - bx[i, j]

        b[2] = bx[i, j] - bx[i, j-1]
        b[3] = bx[i, j+1] - bx[i, j]
        
        b[4] = by[i, j] - by[i-1, j]
        b[5] = by[i+1, j] - by[i, j]

        b[6] = by[i, j] - by[i, j-1]
        b[7] = by[i, j+1] - by[i, j]

        b[8] = bz[i, j] - bz[i-1, j]
        b[9] = bz[i+1, j] - bz[i, j]

        b[10] = bz[i, j] - bz[i, j-1]
        b[11] = bz[i, j+1] - bz[i, j]


        b[12] = self.bx[i, j] - self.bx[i-1, j-1]
        b[13] = self.bx[i+1, j+1] -self.bx[i, j]

        b[14] = self.bx[i, j] - self.bx[i+1, j-1]
        b[15] = self.bx[i-1, j+1] -self.bx[i, j]

        b[16] = self.by[i, j] - self.by[i-1, j-1]
        b[17] = self.by[i+1, j+1] -self.by[i, j]

        b[18] = self.by[i, j] - self.by[i+1, j-1]
        b[19] = self.by[i-1, j+1] -self.by[i, j]

        b[20] = self.bz[i, j] - self.bz[i-1, j-1]
        b[21] = self.bz[i+1, j+1] -self.bz[i, j]

        b[22] = self.bz[i, j] - self.bz[i+1, j-1]
        b[23] = self.bz[i-1, j+1] -self.bz[i, j]

       


        d = np.zeros(24)
        d[0] = -b_derivs[0]*self.dx[i, j]
        d[1] = -d[0]
        
        d[2] = -b_derivs[1]*self.dy[i, j]
        d[3] = -d[2]

        d[4] = -b_derivs[3]*self.dx[i, j]
        d[5] = -d[4]

        d[6] = -b_derivs[4]*self.dy[i, j]
        d[7] = -d[6]

        d[8] = -b_derivs[6]*self.dx[i, j]
        d[9] = -d[8]

        d[10] = -b_derivs[7]*self.dy[i, j]
        d[11] = -d[10]
        
        for di in np.arange(12, 14):
            d[di] = d[di-12] + d[di-10]
        for di in np.arange(14, 16):
            d[di] = -d[di-14] + d[di-12]

        for di in np.arange(16, 18):
            d[di] = d[di-12] + d[di-10]
        for di in np.arange(18, 20):
            d[di] = -d[di-14] + d[di-12]

        for di in np.arange(20, 22):
            d[di] = d[di-12] + d[di-10]
        for di in np.arange(22, 24):
            d[di] = -d[di-14] + d[di-12]
        
        dbx_dz = b_derivs[2]
        dby_dz = b_derivs[5]
        dbz_dz = -b_derivs[0] - b_derivs[4]
        az = np.zeros(24)

        #E-W and N-S
        for di in np.arange(0, 4):
            az[di] = dbx_dz*dz[di]
        do = 4            
        for di in np.arange(0, 4):
            az[di+do] = dby_dz*dz[di]
        do += 4
        for di in np.arange(0, 4):
            az[di+do] = dbz_dz*dz[di]
        do += 4
        #SE-NW and NE-SW
        for di in np.arange(0, 4):
            az[di+do] = dbx_dz*dz[di+4]
        do += 4
        for di in np.arange(0, 4):
            az[di+do] = dby_dz*dz[di+4]
        do += 4
        for di in np.arange(0, 4):
            az[di+do] = dbz_dz*dz[di+4]


        np.testing.assert_almost_equal(loss, loss_expected, 6)

    def test_loss_fn_grad(self):
        
        jmax = 5
        arcsec_per_px = 0.055
        diameter = 20.0
        wavelength = 5250.0
        nx = 10
        defocus = 1.0
        
        gamma = 1.
        L = 3
        prior_prec = np.linspace(0, 1., jmax)**2
        
        psf = psf_basis.psf_basis(jmax = jmax, nx = nx, arcsec_per_px = arcsec_per_px, diameter = diameter, wavelength = wavelength, defocus = defocus, prior_prec=prior_prec)
        psf.create_basis(do_fft=True, do_defocus=True)


        betas = np.random.normal(size=(L, jmax)) + np.random.normal(size=(L, jmax))*1.j
        Ds = np.tile(np.stack((D, D_d)), (L, 1)).reshape((L, 2, nx, nx))
        print("Ds", Ds.shape)
        theta, data = psf.encode(betas, Ds, gamma)

        delta_betas = np.ones_like(betas, dtype='float')*1.0e-8# + betas*1.0e-7

        lik = psf.likelihood(theta, data)
        #print("lik", lik)
        liks = np.tile(lik, (betas.shape[0], betas.shape[1]))
        #print("liks", liks)
        liks1_real = np.zeros_like(betas.real)
        liks1_imag = np.zeros_like(betas.imag)
        for l in np.arange(0, L):
            for i in np.arange(0, betas.shape[1]):
                delta = np.zeros_like(betas)
                delta[l, i] = delta_betas[l, i]
                betas1 = betas+delta
                theta1, _ = psf.encode(betas1, Ds, gamma)
                
                liks1_real[l, i] = psf.likelihood(theta1, data)
    
                delta[l, i] = 1.j*delta_betas[l, i]
                betas1 = betas+delta
                theta1, _ = psf.encode(betas1, Ds, gamma)
    
                liks1_imag[l, i] = psf.likelihood(theta1, data)
        
        #print((liks1_real - liks) / delta_betas.real, (liks1_imag - liks) / delta_betas.imag)
        #print(liks1_real)
        grads_expected = np.stack(((liks1_real - liks) / delta_betas, (liks1_imag - liks) / delta_betas), axis=1).flatten()
    
        grads = psf.likelihood_grad(theta, data)

        np.testing.assert_array_almost_equal(grads, grads_expected, 1)
   '''         

        
if __name__ == '__main__':
    unittest.main()
