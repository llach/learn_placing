import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):

    def __init__(self, base, head, mutation_scale=20, lw=3, arrowstyle="-|>", color="r", **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), mutation_scale=mutation_scale, lw=lw, arrowstyle=arrowstyle, color=color, **kwargs)
        self._verts3d = list(zip(base,head))

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

class AxesPlot:

    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(9.71, 8.61))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.clean()
    
    def clean(self):
        self.ax.clear()

        alim = [-1.2, 1.2]
        self.ax.set_xlim(alim)
        self.ax.set_ylim(alim)
        self.ax.set_zlim(alim)

        aalph = 0.9
        self.ax.add_artist(Arrow3D([0,0,0], [1,0,0], color=[1.0, 0.0, 0.0, aalph]))
        self.ax.add_artist(Arrow3D([0,0,0], [0,1,0], color=[0.0, 1.0, 0.0, aalph]))
        self.ax.add_artist(Arrow3D([0,0,0], [0,0,1], color=[0.0, 0.0, 1.0, aalph]))

        self.handles = []
        self.handles.append(
            self.ax.add_artist(Arrow3D([0,0,0], [0,0,-1], color=[0.0, 1.0, 1.0, 0.7], label=f"desired normal"))
        )
    
    def plot_v(self, vec, start=[0,0,0], color="blue", label="", legend=False):
        h = self.ax.add_artist(Arrow3D(start, vec, color=color, label=label))
        if legend or label!="": self.handles.append(h)

    def plot_points(self, points, *args, **kwargs):
        h = self.ax.scatter(points[:,0], points[:,1], points[:,2], *args, **kwargs)
        if "label" in kwargs: self.handles.append(h)

    def title(self, t):
        self.fig.suptitle(t)

    def show(self):
        self.ax.legend(handles=self.handles)
            
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show()