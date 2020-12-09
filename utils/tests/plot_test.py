import sys
sys.path.append('..')
import plot
import matplotlib.pyplot as plt
import unittest


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
        
if __name__ == '__main__':
    unittest.main()
