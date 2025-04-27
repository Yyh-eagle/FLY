import serial
import struct


    
    
#串口通信类,保存所有全局变量，并且作为参数可以四处传递
class SerialPort():
    def __init__(self):

        self.serial_port = serial.Serial(
            port='/dev/ttyUSB0',#串口号#bug 固定串口
            baudrate=460800,#波特率
            bytesize=serial.EIGHTBITS,#八位字节
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        self.init_receive_var()
        self.init_send_var()


    #初始化接受变量
    def init_receive_var(self):
        self.ifArrive_int = 0#是否飞到指定目标，镜头开始工作
        self.task_id_int =0#是否完成当前任务
        self.task_state_int = 0#任务状态机控制
        self.d435_yaw_float =0.0#舵机的角度反
        self.Task_data3_float =None
        self.Task_data4_float = None
        self.Task_data5_float = None
        self.Task_data6_float = None
        self.Task_data7_int = None
        self.Task_data8_int = None
        self.receive_num = None
        
    #接收动作，每个周期执行
    def receive(self):

  
        response =None
        size = self.serial_port.inWaiting()
        
        if  size>0:
            response = self.serial_port.read(44)
            
        if response is not None:
            if response[0]==179 and  response[1]==179 :#帧头校验
                self.parse_packet(response)

        self.serial_port.flushInput()
        self.receive_num = [self.ifArrive_int,self.task_id_int,self.task_state_int,self.d435_yaw_float,self.Task_data3_float,self.Task_data4_float,self.Task_data5_float,self.Task_data6_float,self.Task_data7_int,self.Task_data8_int]
    #数据解包分配
    def parse_packet(self,packet):
        # 提取数据部分（去除帧头和帧尾）0
        data = packet[2:-2]
        
        # 解析各浮点数字段（小端模式）
        self.ifArrive_int = struct.unpack('<i', data[0:4])[0]
        self.task_id_int = struct.unpack('<i', data[4:8])[0]
        self.task_state_int = struct.unpack('<i', data[8:12])[0]
        self.d435_yaw_float = struct.unpack('<f', data[12:16])[0]
        self.Task_data3_float = struct.unpack('<f', data[16:20])[0]
        self.Task_data4_float = struct.unpack('<f', data[20:24])[0]
        self.Task_data5_float = struct.unpack('<f', data[24:28])[0]
        self.Task_data6_float = struct.unpack('<f', data[28:32])[0]
        self.Task_data7_int = struct.unpack('<i', data[32:36])[0]
        self.Task_data8_int = struct.unpack('<i', data[36:40])[0]
    #------------------------------------------数据发送----------------------------------------------------#
    #------------------------------------------一共19个浮点数，外加4位，共有71个u8--------------------------#    
    def init_send_var(self):#所有单位都是厘米
        self.t_flag_u = 0#T265是否工作
        self.d_flag_u = 0#D435i是否检测到目标
        self.c_flag_u = 0#小相机是否检测到目标
        self.D435_aim_i  =0#d435i检测目标类别
        self.c_aim_i = 0#小相机检测目标类别
        self.T265_x_f = 0
        self.T265_y_f = 0
        self.T265_z_f = 0
        self.D435_x_f = 0
        self.D435_y_f = 0
        self.D435_z_f = 0#12
        self.c_x_f = 0#小相机的cm目标
        self.c_y_f = 0
        self.send_data1_f = 0
        self.send_data2_f = 0
        self.send_data3_f = 0
        self.send_data4_f = 0#18
        self.send_data5_i = 0
        self.send_data6_i = 6
        

    #每个周期循环一次
    def Send_message(self):
        
        self.data_num = [self.t_flag_u,self.d_flag_u,self.c_flag_u,\
                        self.D435_aim_i,self.c_aim_i,\
                        self.T265_x_f,self.T265_y_f,self.T265_z_f,self.D435_x_f,self.D435_y_f,self.D435_z_f,self.c_x_f,self.c_y_f,\
                        self.send_data1_f,self.send_data2_f,self.send_data3_f,self.send_data4_f,self.send_data5_i,self.send_data6_i]
    
        transdata = [0xb3,0xb3,self.t_flag_u,self.d_flag_u,self.c_flag_u]#待发送的数据
        for data in self.data_num[3:5]:
            
            data_bytes = struct.pack('i', data)
            transdata.extend(data_bytes)

        for data in self.data_num[5:17]:
            
            data_bytes = struct.pack('f', data)
            transdata.extend(data_bytes)
            
        for data in self.data_num[17:]:
            
            data_bytes = struct.pack('i', data)
            transdata.extend(data_bytes)
        
        # 追加结束标志
        transdata.extend([0x5b, 0x5b])
        print(f"{transdata=}")
        byte_data = bytearray(transdata)
        self.serial_port.write(byte_data)
    

    def Timer_zero(self):#每个定时器周期开始前需要归零的数据
        self.t_flag_u = 0#T265是否工作
        self.d_flag_u = 0#D435i是否检测到目标
        self.c_flag_u = 0#小相机是否检测到目标
        self.D435_aim_i  =0#d435i检测目标类别
        self.c_aim_i = 0#小相机检测目标类别

  

#----------------------------------------测试代码-----------------------------------------------#
ser = SerialPort()
ind = 0
while 1:
    ind += 1
    ser.receive()

    ser.Send_message()

    if(ind>100):
        break