import time
import numpy
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

x = numpy.linspace(-2 * numpy.pi, 2 * numpy.pi, 1000)
y = numpy.cos(x)

# Plot
win = pg.GraphicsWindow()
win.resize(800, 800)

p = win.addPlot()
scatter = pg.ScatterPlotItem(pxMode = False)
p.addItem(scatter)

i = 0
while i < 5000:
  start_time = time.time()
  items = [{'pos': (0.0, 2.5), 'size': 1e-2}]

  noise = numpy.random.normal(0, 1, len(y))
  y_new = y + noise
  scatter.setData(items)
#   p.plot(x, y_new, pen = "y", clear = True)
#   p.enableAutoRange("xy", False)

  pg.QtGui.QApplication.processEvents()

  i += 1

  end_time = time.time()
  print("It has been {0} seconds since the loop started".format(end_time - start_time))

win.close()
