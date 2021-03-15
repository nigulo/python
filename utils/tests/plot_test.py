import sys
sys.path.append('..')
import plot
import matplotlib.pyplot as plt
import unittest
import numpy as np


class test_plot(unittest.TestCase):
    
    def test_rectangle(self):
        myplot = plot.plot()
        #myplot.line(-1, -2, 3, 4)
        myplot.rectangle(.1, .2, .3, .4, fill=True, facecolor="blue", alpha=.5)
        myplot.save("rectangle.png")
        myplot.close()

    def test_colormap(self):
        image = plt.imread("gradient.png")
        myplot = plot.plot()
        #if len(image.shape) == 3:
        #    image = image[:, :, 0]
        myplot.colormap(image)#, vmin=min_val, vmax=max_val)
        myplot.save("colormap.png")
        myplot.close()

        data = np.empty((100, 100))
        val = 0
        for y in range(100):
            for x in range(100):
                data[y, x] = val
            val += 1

        myplot = plot.plot()
        myplot.colormap(data)#, vmin=min_val, vmax=max_val)
        myplot.save("colormap2.png")
        myplot.close()
        
if __name__ == '__main__':
    unittest.main()
