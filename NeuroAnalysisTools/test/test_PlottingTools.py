import os, unittest
import NeuroAnalysisTools.core.PlottingTools as pt

class TestTimingAnalysis(unittest.TestCase):

    def setUp(self):
        pass

    def test_density_scatter_plot2(self):
        import matplotlib.pyplot as plt

        pt.density_scatter_plot2(x=[3, 4], y=[5, 6], diffusion_constant=0.5, pixel_res=30,
                                 std_lim=3)
        plt.show()

        pt.density_scatter_plot2(x=[3, 4], y=[2, 50], diffusion_constant=0.5, pixel_res=31)
        plt.show()

    def test_density_scatter_plot3(self):

        import numpy as np
        import matplotlib.pyplot as plt

        m1 = np.random.normal(size=1000)
        m2 = np.random.normal(scale=0.5, size=1000)

        x, y = m1 + m2, m1 - m2

        pt.density_scatter_plot3(x=x, y=y, pixel_res=30)
        plt.show()

        pt.density_scatter_plot3(x=[3, 4], y=[5, 6], pixel_res=50, cmap='magma',
                                 bandwidth=0.1)
        plt.show()

