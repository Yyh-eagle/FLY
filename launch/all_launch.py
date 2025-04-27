import os
 
from ament_index_python.packages import get_package_share_directory
 
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
 
 
def generate_launch_description():
   tf_node = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('yyh_nav'), 'launch'),
         '/tf_launch.py'])
      )
   realsense_node = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('realsense2_camera'), 'launch'),
         '/rs_d435i_and_t265_launch.py'])
      )
   image_node = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('yyh_object'), 'launch'),
         '/image_launch.py'])
      )
   
   return LaunchDescription([
      tf_node,
      realsense_node,
      image_node
   ])
