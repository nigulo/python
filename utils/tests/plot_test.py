import sys
sys.path.append('..')
import plot
import unittest


class test_plot(unittest.TestCase):
    
    def test_rectangle(self):
        myplot = plot.plot()
        #myplot.line(-1, -2, 3, 4)
        myplot.rectangle(.1, .2, .3, .4, fill=True, facecolor="blue", alpha=.5)
        myplot.save("rectangle.png")
        myplot.close()

        
if __name__ == '__main__':
    unittest.main()
