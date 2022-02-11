import unittest

from summaries.analysis import DensityPlot


class TestDensityPlot(unittest.TestCase):

    def test_find_closest_reference_match(self):
        plot = DensityPlot()

        extracted = "Das ist ein Test."
        reference = ["Ein längerer Inhalt.", "Er besteht aus mehreren Tests.", "Das ist ein Test."]

        self.assertEqual(2/3, plot.find_closest_reference_match(extracted, reference))


if __name__ == '__main__':
    unittest.main()
