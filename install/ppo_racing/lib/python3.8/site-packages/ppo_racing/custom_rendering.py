import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
import numpy as np

class CarTrackRecorder:
    def __init__(self, track_image_path, track_extent, raceline_waypoints=None):
        self.track_extent = track_extent
        self.frames = []
        self.triangle_size = 1.5

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(track_extent[0], track_extent[1])
        self.ax.set_ylim(track_extent[2], track_extent[3])
        self.ax.set_aspect('equal')

        # Load image
        self.track_img = mpimg.imread(track_image_path)
        self.ax.imshow(self.track_img, extent=track_extent)

        # Raceline
        if raceline_waypoints:
            wp = np.array(raceline_waypoints)
            self.ax.plot(wp[:, 0], wp[:, 1], 'y--')

        # Car triangle template
        self.triangle_shape = np.array([
            [0, self.triangle_size],
            [-self.triangle_size / 2, -self.triangle_size],
            [self.triangle_size / 2, -self.triangle_size]
        ])

        self.car_patches = []
        for color in ['red', 'blue']:
            patch = patches.Polygon(self.triangle_shape, closed=True, color=color)
            self.ax.add_patch(patch)
            self.car_patches.append(patch)

    def draw_frame(self, car_states):
        for patch, (x, y, yaw) in zip(self.car_patches, car_states):
            trans = Affine2D().rotate_around(0, 0, yaw).translate(x, y) + self.ax.transData
            patch.set_transform(trans)

        # Draw and store as frame
        self.fig.canvas.draw()
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(image)

    def save_video(self, path="videos/output.mp4", fps=20):
        import imageio
        imageio.mimsave(path, self.frames, fps=fps)
