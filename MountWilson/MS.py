import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt

###############################################################################
# Load rotational periods
try:
    from itertools import izip_longest  # added in Py 2.6
except ImportError:
    from itertools import zip_longest as izip_longest  # name change in Py 3.x

try:
    from itertools import accumulate  # added in Py 3.2
except ImportError:
    def accumulate(iterable):
        'Return running totals (simplified version).'
        total = next(iterable)
        yield total
        for value in iterable:
            total += value
            yield total

def make_parser(fieldwidths):
    cuts = tuple(cut for cut in accumulate(abs(fw) for fw in fieldwidths))
    pads = tuple(fw < 0 for fw in fieldwidths) # bool values for padding fields
    flds = tuple(izip_longest(pads, (0,)+cuts, cuts))[:-1]  # ignore final one
    parse = lambda line: tuple(line[i:j] for pad, i, j in flds if not pad)
    # optional informational function attributes
    parse.size = sum(abs(fw) for fw in fieldwidths)
    parse.fmtstring = ' '.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's')
                                                for fw in fieldwidths)
    return parse


stars = list()
bmvs = list()
ms = list()

fieldwidths = (10, 8, 10, 6, 8, 4)  # negative widths represent ignored padding fields
parse = make_parser(fieldwidths)
line_no = 0
#with open(path+"mwo-rhk.dat", "r") as ins:
with open("mwo-hr.dat", "r") as ins:
    for line in ins:
        if line_no < 19:
            line_no += 1
            continue
        fields = parse(line)
        star = fields[0].strip()
        star = star.replace(' ', '')
        star = star.upper()
        try:
            v = float(fields[3].strip())
        except ValueError:
            v = None
        try:
            bmv = float(fields[4].strip())
        except ValueError:
            bmv = None
        try:
            d = float(fields[5].strip())
        except ValueError:
            d = None
        if v is not None and bmv is not None and d is not None:
            a = 1.5/1000
            m = v - 5 * np.log10(0.1*d) - a*d
            stars.append(star)
            bmvs.append(bmv)
            ms.append(m)


msbv = [0.15,0.30,0.35,0.44,0.52,0.58,0.63,0.68,0.74,0.81,0.91,1.15,1.40,1.49,1.64]
msm = [1.95,2.7,3.1,3.5,4.0,4.4,4.7,5.1,5.5,5.9,6.4,7.35,8.8,9.9,12.3]
msf = inter.interp1d(msbv, msm, kind = 'cubic')

stars = np.asarray(stars)
bmvs = np.asarray(bmvs)
ms = np.asarray(ms)

def determine(i):
    bmv = bmvs[i] 
    m = ms[i]
    m_v_ms = msf(bmv)
    if (m - m_v_ms < -1.0):
        return False
    return True

indices = [i for i in np.arange(0, len(stars)) if determine(i)]

stars = stars[indices]
bmvs = bmvs[indices]
ms = ms[indices]

bmvs_dense = np.linspace(min(msbv), max(msbv), 1000)
plt.plot(bmvs, ms, 'o', bmvs_dense, msf(bmvs_dense), '-')
plt.show()


