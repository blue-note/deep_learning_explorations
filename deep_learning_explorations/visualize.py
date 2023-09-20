from manim import *
from manim.camera.camera import Camera

class NeuralNetworkScene(ThreeDScene):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def construct(self):
        # Create input layer
        input_layer = [Dot(np.array([i, 2, 0])) for i in range(2)]
        input_label = Text("Input Layer").next_to(input_layer[0], UP)

        # Create hidden layer
        hidden_layer = [Dot(np.array([i, 0, 0])) for i in range(3)]
        hidden_label = Text("Hidden Layer").next_to(hidden_layer[0], UP)

        # Create output layer
        output_layer = [Dot(np.array([i, -2, 0])) for i in range(2)]
        output_label = Text("Output Layer").next_to(output_layer[0], UP)

        # Connect the layers with lines
        for start_dot in input_layer:
            for end_dot in hidden_layer:
                self.add(Line(start_dot.get_center(), end_dot.get_center()))

        for start_dot in hidden_layer:
            for end_dot in output_layer:
                self.add(Line(start_dot.get_center(), end_dot.get_center()))

        # Display everything
        self.add(*input_layer, *hidden_layer, *output_layer, input_label, hidden_label, output_label)

class InputPointVisualization(Scene):
    def __init__(self, points, labels):
        super().__init__()
        self.points = points
        self.labels = labels

    def construct(self):
        # Add 2D axes
        axes = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            axis_config={"color": WHITE},
        )
        self.add(axes)
        # Iterate over points and labels to create Dots and add them to the scene
        for point, label in zip(self.points, self.labels):
            color = BLUE if label == 1 else YELLOW
            point_3d = np.append(point, [0])
            dot = Dot(point=point_3d, color=color).scale(0.3)  # scale to make dots smaller
            self.add(dot)

        self.wait(1)


class IntermediatePointVisualization(ThreeDScene):
    def __init__(self, input_points, input_labels, intermediate_points):
        super().__init__()
        self.input_points = input_points
        self.input_labels = input_labels
        self.intermediate_points = intermediate_points

    def construct(self):
        # Set the camera position
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        axes = ThreeDAxes()
        self.add(axes)

        # Iterate over points and labels to create Dots and add them to the scene
        for point, label in zip(self.intermediate_points, self.input_labels):
            color = BLUE if label == 1 else YELLOW  
            dot = Dot3D(point=point, color=color).scale(0.2)  # scale to make dots smaller
            self.add(dot)

        self.wait(1)


class OutputPointVisualization(Scene):
    def __init__(self, input_labels, output_points):
        super().__init__()
        self.input_labels = input_labels
        self.output_points = output_points

    def construct(self):
        # Iterate over output points and input labels to create Dots and add them to the scene
        for point, label in zip(self.output_points, self.input_labels):
            color = BLUE if label == 1 else YELLOW
            point_3d = np.append(point, [0])  # Convert 2D point to 3D by appending a 0 for the z-axis
            dot = Dot(point=point_3d, color=color).scale(0.1)  # scale to make dots smaller
            self.add(dot)

        self.wait(1)
