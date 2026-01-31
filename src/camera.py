import bpy
import math
from mathutils import Euler

# Center position
# center = (8.86993, -0.88, 3.99213) # bed
center = (7.84387, -0.88, -3.82026)  # tv

num = 40

# List of radii to create
radius_list = [2.50, 2.25, 2.00, 1.75, 1.50, 1.25, 1, 0.75]

# Create cameras
for radius in radius_list:
    for i in range(5, 20):  # Only create cameras with IDs 6 to 20
        # Calculate camera position
        angle = (2 * math.pi / num) * i
        pos_x = center[0] + radius * math.cos(angle)
        pos_z = center[2] + radius * math.sin(angle)

        # Create camera object
        bpy.ops.object.camera_add(location=(pos_x, -0.88, pos_z))
        camera = bpy.context.object

        # Set camera parameters
        camera.data.angle = math.radians(90)  # Field of view is 90 degrees
        camera.rotation_euler = Euler((0, math.radians(i * 360 / num - 90), math.radians(180)),
                                      'XYZ')  # Set Euler angles

        # Set camera name
        camera.name = f"Camera_r{radius}_num{i + 1}"