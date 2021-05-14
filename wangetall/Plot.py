import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg


class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        #### Create Gui Elements ###########
        pg.setConfigOption('background', 'w')

        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100) # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)


        # self.view = self.canvas.addViewBox()
        # self.view.setAspectLocked(True)
        # self.view.setRange(QtCore.QRectF(0,0, 100, 100))

        #  image plot
        # self.img = pg.ImageItem(border='w')
        # self.view.addItem(self.img)

        # self.canvas.nextRow()
        #  line plot
        self.otherplot = self.canvas.addPlot()
        self.h2 = self.otherplot.plot(pen=None)


        #### Set Data  #####################

        self.x = np.linspace(0,50., num=100)
        self.X,self.Y = np.meshgrid(self.x,self.x)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        # self._update()
    def onNewData(self, tracker):
        static_background_state = tracker.state.static_background
        
        points_dict_list = []
        # ax.scatter(static_background_state.xb[:,0], static_background_state.xb[:,1], color="black", label="Static Background", s=20)
        for i in range(len(static_background_state.xb[:,0])):
            points_dict = {"pos": (static_background_state.xb[i,0], static_background_state.xb[i,1]),
                                        "size": 0.2, 'pen': {'color': 'b', 'width': 2}, 'brush': pg.intColor(10, 100)}
            points_dict_list.append(points_dict)
        
        points_dict = {"pos": (tracker.state.xs[0], tracker.state.xs[1],),
                            "size": 0.2, 'pen': {'color': 'b', 'width': 2}, 'brush': pg.intColor(10, 100)}
        points_dict_list.append(points_dict)
        # ax.scatter(tracker.state.xs[0], tracker.state.xs[1], color="blue")
        # lidararc = Arc((tracker.state.xs[0], tracker.state.xs[1]), \
        #     width = 2, height = 2,\
        #     angle = math.degrees(tracker.state.xs[2]),\
        #     theta1= math.degrees(-4.7/2), theta2 = math.degrees(4.7/2), color="turquoise", linestyle="--",  alpha=0.8)
        # ax.add_patch(lidararc)

        for idx, track in tracker.state.dynamic_tracks.items():
            points_dict = {"pos": (track.kf.x[0], track.kf.x[1],),
                            "size": 0.2, 'pen': {'color': 'purple', 'width': 5}, 'brush': pg.intColor(10, 100)}
            points_dict_list.append(points_dict)

            # ax.scatter(track.kf.x[0], track.kf.x[1], color="purple", s=60)
            # ax.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], s = 1, c = track.color)
            
            for i in range(len(track.xp[:,0])):
                points_dict = {"pos": (track.xp[i,0]+track.kf.x[0], track.xp[i,1]+track.kf.x[1]),
                            "size": 0.2, 'pen': {'color': track.color, 'width': 2}, 'brush': pg.intColor(10, 100)}
                points_dict_list.append(points_dict)

            # trackspeed = round(np.sqrt(track.kf.x[3]**2+track.kf.x[4]**2), 2)                    
            # if track.parent is not None:
            #     ax.text(track.kf.x[0], track.kf.x[1], "T{} S:{}, P:{}".format(idx, trackspeed, track.parent), size = "x-small")
            # else:
            #     ax.text(track.kf.x[0], track.kf.x[1], "T{} S:{}".format(idx, trackspeed), size = "x-small")
            # if track.id ==  13 or track.id == 24 or track.id == 131:
            #     tracked_object_speeds[i,1:3] = [track.id, trackspeed]
            self.setData(points_dict_list)
    def setData(self, ls):
        self.h2.setData(ls)

    def _update(self, tracker):
        # self.otherplot.setXRange(tracker.state.xs[0] - 10, tracker.state.xs[0] + 10)
        # self.otherplot.setYRange(tracker.state.xs[1] - 10, tracker.state.xs[1] + 10)

        static_background_state = tracker.state.static_background
        
        points_dict_list = []
        # ax.scatter(static_background_state.xb[:,0], static_background_state.xb[:,1], color="black", label="Static Background", s=20)
        for i in range(len(static_background_state.xb[:,0])):
            points_dict = {"pos": (static_background_state.xb[i,0], static_background_state.xb[i,1]),
                                        "size": 0.2, 'pen': {'color': 'b', 'width': 2}, 'brush': pg.intColor(10, 100)}
            points_dict_list.append(points_dict)
        
        points_dict = {"pos": (tracker.state.xs[0], tracker.state.xs[1],),
                            "size": 0.2, 'pen': {'color': 'b', 'width': 2}, 'brush': pg.intColor(10, 100)}
        points_dict_list.append(points_dict)
        # ax.scatter(tracker.state.xs[0], tracker.state.xs[1], color="blue")
        # lidararc = Arc((tracker.state.xs[0], tracker.state.xs[1]), \
        #     width = 2, height = 2,\
        #     angle = math.degrees(tracker.state.xs[2]),\
        #     theta1= math.degrees(-4.7/2), theta2 = math.degrees(4.7/2), color="turquoise", linestyle="--",  alpha=0.8)
        # ax.add_patch(lidararc)

        for idx, track in tracker.state.dynamic_tracks.items():
            points_dict = {"pos": (track.kf.x[0], track.kf.x[1],),
                            "size": 0.2, 'pen': {'color': 'purple', 'width': 5}, 'brush': pg.intColor(10, 100)}
            points_dict_list.append(points_dict)

            # ax.scatter(track.kf.x[0], track.kf.x[1], color="purple", s=60)
            # ax.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], s = 1, c = track.color)
            
            for i in range(len(track.xp[:,0])):
                points_dict = {"pos": (track.xp[i,0]+track.kf.x[0], track.xp[i,1]+track.kf.x[1]),
                            "size": 0.2, 'pen': {'color': track.color, 'width': 2}, 'brush': pg.intColor(10, 100)}
                points_dict_list.append(points_dict)

            # trackspeed = round(np.sqrt(track.kf.x[3]**2+track.kf.x[4]**2), 2)                    
            # if track.parent is not None:
            #     ax.text(track.kf.x[0], track.kf.x[1], "T{} S:{}, P:{}".format(idx, trackspeed, track.parent), size = "x-small")
            # else:
            #     ax.text(track.kf.x[0], track.kf.x[1], "T{} S:{}".format(idx, trackspeed), size = "x-small")
            # if track.id ==  13 or track.id == 24 or track.id == 131:
            #     tracked_object_speeds[i,1:3] = [track.id, trackspeed]

        self.data = np.sin(self.X/3.+self.counter/9.)*np.cos(self.Y/3.+self.counter/9.)
        self.ydata = np.sin(self.x/3.+ self.counter/9.)

        # self.img.setImage(self.data)
        # self.h2.setData(self.ydata)
        self.h2.setData(points_dict_list)
        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
