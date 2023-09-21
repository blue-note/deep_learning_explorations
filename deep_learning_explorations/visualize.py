from typing import List
from manim import *
from manim.camera.camera import Camera


class NetworkVisualization(ThreeDScene):
    """
    input_points: a list of 2D points
    intermediate_points: a list of lists of 2D points, one list for each hidden layer
    output_probabilities: a list of 2D points from the output layer
    """
    def __init__(self, input_points, input_labels, intermediate_outputs, output_probabilities, **kwargs):
        super().__init__(**kwargs)
        self.input_points = input_points
        self.input_labels = input_labels
        self.intermediate_outputs = intermediate_outputs
        self.output_probabilities = output_probabilities

    def construct(self):
        mobjects = []

        # Create the axes
        axes = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            axis_config={"color": WHITE},
        )
        mobjects.append(axes)
        self.play(Write(axes))

        # Create input dots
        input_dots = self.input_point_visualization(self.input_points, self.input_labels)
        self.play(*[Write(dot) for dot in input_dots])

        for dot in input_dots:
            self.remove(dot)

        # Create intermediate dots
        for intermediate_points in self.intermediate_outputs:
            intermediate_dots = self.intermediate_point_visualization(self.input_labels, intermediate_points)
            self.play(*[Write(dot) for dot in intermediate_dots])
            for dot in intermediate_dots:
                self.remove(dot)

        # Create output dots
        output_dots = self.output_point_visualization(self.input_labels, self.output_probabilities)
        self.play(*[Write(dot) for dot in output_dots])

    def input_point_visualization(self, points, labels):
        dots = []
        for point, label in zip(points, labels):
            color = BLUE if label == 1 else YELLOW
            point_3d = np.append(point, [0])
            dot = Dot(point=point_3d, color=color).scale(0.3)  # scale to make dots smaller
            dots.append(dot)
        return dots

    def intermediate_point_visualization(self, input_labels, intermediate_points):
        dots = []
        for point, label in zip(intermediate_points, input_labels):
            color = BLUE if label == 1 else YELLOW  
            dot = Dot(point=point, color=color).scale(0.2)
            dots.append(dot)
        return dots

    def output_point_visualization(self, input_labels, output_points):
        dots = []
        for point, label in zip(output_points, input_labels):
            color = BLUE if label == 1 else YELLOW
            point_3d = np.append(point, [0])  # Convert 2D point to 3D by appending a 0 for the z-axis
            dot = Dot(point=point_3d, color=color).scale(0.3)  # scale to make dots smaller
            dots.append(dot)
        return dots

# class InputPointVisualization(Scene):
#     def __init__(self, points, labels):
#         super().__init__()
#         self.points = points
#         self.labels = labels

#     def construct(self):
#         mobjects = []
#         # Add 2D axes
#         axes = Axes(
#             x_range=[-4, 4],
#             y_range=[-4, 4],
#             axis_config={"color": WHITE},
#         )
#         mobjects.append(axes)
#         # Iterate over points and labels to create Dots and add them to the scene
#         for point, label in zip(self.points, self.labels):
#             color = BLUE if label == 1 else YELLOW
#             point_3d = np.append(point, [0])
#             dot = Dot(point=point_3d, color=color).scale(0.3)  # scale to make dots smaller
#             mobjects.append(dot)

#         self.wait(1)


# class IntermediatePointVisualization(ThreeDScene):
#     def __init__(self, input_points, input_labels, intermediate_points):
#         super().__init__()
#         self.input_points = input_points
#         self.input_labels = input_labels
#         self.intermediate_points = intermediate_points

#     def construct(self):
#         # Set the camera position
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
#         axes = ThreeDAxes()
#         self.add(axes)

#         # Iterate over points and labels to create Dots and add them to the scene
#         for point, label in zip(self.intermediate_points, self.input_labels):
#             color = BLUE if label == 1 else YELLOW  
#             dot = Dot3D(point=point, color=color).scale(0.2)  # scale to make dots smaller
#             self.add(dot)

#         self.wait(1)


# class OutputPointVisualization(Scene):
#     def __init__(self, input_labels, output_points):
#         super().__init__()
#         self.input_labels = input_labels
#         self.output_points = output_points

#     def construct(self):
#         # Iterate over output points and input labels to create Dots and add them to the scene
#         for point, label in zip(self.output_points, self.input_labels):
#             color = BLUE if label == 1 else YELLOW
#             point_3d = np.append(point, [0])  # Convert 2D point to 3D by appending a 0 for the z-axis
#             dot = Dot(point=point_3d, color=color).scale(0.1)  # scale to make dots smaller
#             self.add(dot)

#         self.wait(1)

