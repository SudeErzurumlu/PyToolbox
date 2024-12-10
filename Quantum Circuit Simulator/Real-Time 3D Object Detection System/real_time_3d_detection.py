import numpy as np
import open3d as o3d
from keras.models import load_model

class ObjectDetector3D:
    def __init__(self, model_path):
        """
        Initializes the 3D object detector with a pre-trained deep learning model.
        """
        self.model = load_model(model_path)

    def load_point_cloud(self, file_path):
        """
        Loads a point cloud from a LiDAR data file.
        """
        return o3d.io.read_point_cloud(file_path)

    def preprocess(self, point_cloud):
        """
        Prepares the point cloud data for the model.
        """
        points = np.asarray(point_cloud.points)
        return points / np.linalg.norm(points, axis=1, keepdims=True)

    def detect_objects(self, point_cloud):
        """
        Detects objects in the point cloud using the model.
        """
        preprocessed_data = self.preprocess(point_cloud)
        predictions = self.model.predict(preprocessed_data)
        return predictions

    def visualize(self, point_cloud, detections):
        """
        Visualizes the point cloud with detected objects highlighted.
        """
        for obj in detections:
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=obj[:3], max_bound=obj[3:])
            box.color = (1, 0, 0)
            point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
            o3d.visualization.draw_geometries([point_cloud, box])

# Example Usage:
# detector = ObjectDetector3D("model.h5")
# pc = detector.load_point_cloud("sample.ply")
# detections = detector.detect_objects(pc)
# detector.visualize(pc, detections)
