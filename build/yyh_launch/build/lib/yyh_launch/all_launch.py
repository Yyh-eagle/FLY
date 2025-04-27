from launch import LaunchDescription           # launch文件的描述类
from launch_ros.actions import Node            # 节点启动的描述类
import os
from launch.substitutions import FindPackageShare

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():             # 自动生成launch文件的函数
    
    realsense_share = FindPackageShare('realsense2_camera').find('realsense2_camera')
    realsense_file_path = os.path.join(realsense_share, 'launch', 'rs_d435i_and_t265_launch.py ')
    realsense_file = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_file_path),
    )
    my_share = FindPackageShare('yyh_launch').find('yyh_launch')
    my_file_path = os.path.join(my_share, 'launch', 'my_launch.py ')
    my_file = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(my_file_path),
    )
    
    
    LD = LaunchDescription()
    LD.add_action(realsense_file)
    LD.add_action(my_file)

    return LD    