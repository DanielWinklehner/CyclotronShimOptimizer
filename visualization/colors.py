from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator

__author__ = "Daniel Winklehner"


class MyColors(object):

    def __init__(self):
        """
        Constructor
        """
        self.colors = ['#4B82B8',
                       '#B8474D',
                       '#95BB58',
                       '#234B7C',
                       '#8060A9',
                       '#53A2CB',
                       '#FC943B']

        _numerals = '0123456789abcdefABCDEF'
        self._hexdec = {v: int(v, 16) for v in (x + y for x in _numerals for y in _numerals)}

    def __getitem__(self, item):

        return self.colors[int(item % 7)]

    def get_in_rgb(self, item):
        # Source: https://stackoverflow.com/questions/4296249/how-do-i-convert-a-hex-triplet-to-an-rgb-tuple-and-back
        triplet = self.colors[int(item % 7)][1:]
        return [self._hexdec[triplet[0:2]] / 255.0,
                self._hexdec[triplet[2:4]] / 255.0,
                self._hexdec[triplet[4:6]] / 255.0, 255.0 / 255.0]

    def show_in_pyplot(self):
        for i in range(7):
            plt.plot([0, 1], [i, i], color=self[i], label="Index {}".format(i))
        plt.title("Colors available from colors.py")
        plt.ylabel("Index")
        plt.gca().get_xaxis().set_major_locator(LinearLocator(numticks=0))
        plt.show()


if __name__ == "__main__":
    col = MyColors()
    col.show_in_pyplot()
