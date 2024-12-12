import numpy as np
import matplotlib.pyplot as plt

class RayTracer:
    def __init__(self, width, height, fov):
        """
        Initializes the ray tracer with screen dimensions and field of view.
        """
        self.width = width
        self.height = height
        self.fov = np.deg2rad(fov)
        self.scene = []
        self.camera = np.array([0, 0, 0])

    def add_sphere(self, center, radius, color):
        """
        Adds a sphere to the scene.
        """
        self.scene.append({"center": np.array(center), "radius": radius, "color": np.array(color)})

    def render(self):
        """
        Renders the 3D scene using ray tracing.
        """
        image = np.zeros((self.height, self.width, 3))
        for y in range(self.height):
            for x in range(self.width):
                ray_dir = self.compute_ray_direction(x, y)
                color = self.trace_ray(self.camera, ray_dir)
                image[y, x] = color
        return np.clip(image, 0, 1)

    def compute_ray_direction(self, x, y):
        """
        Computes the direction of a ray for a pixel.
        """
        aspect_ratio = self.width / self.height
        screen_x = (2 * (x + 0.5) / self.width - 1) * np.tan(self.fov / 2) * aspect_ratio
        screen_y = (1 - 2 * (y + 0.5) / self.height) * np.tan(self.fov / 2)
        return np.array([screen_x, screen_y, -1])

    def trace_ray(self, origin, direction):
        """
        Traces a ray and calculates its color based on intersections.
        """
        closest_t = float("inf")
        hit_color = np.array([0, 0, 0])
        for obj in self.scene:
            t = self.intersect_sphere(origin, direction, obj["center"], obj["radius"])
            if t and t < closest_t:
                closest_t = t
                hit_color = obj["color"]
        return hit_color

    def intersect_sphere(self, origin, direction, center, radius):
        """
        Determines if a ray intersects a sphere.
        """
        oc = origin - center
        b = 2 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius**2
        discriminant = b**2 - 4 * c
        if discriminant < 0:
            return None
        return (-b - np.sqrt(discriminant)) / 2

# Example Usage
rt = RayTracer(400, 300, 90)
rt.add_sphere([0, 0, -5], 1, [1, 0, 0])  # Red sphere
rt.add_sphere([2, 0, -6], 1, [0, 1, 0])  # Green sphere
image = rt.render()
plt.imshow(image)
plt.show()
