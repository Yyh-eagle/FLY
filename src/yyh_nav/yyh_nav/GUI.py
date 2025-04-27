import tkinter as tk
from tkinter import ttk

class DebugGUI(tk.Tk):
    def __init__(self, subscriber_node):
        super().__init__()
        self.sub_node = subscriber_node
        self.title("变量监控界面")
        self.geometry("300x200")
        
        self.create_widgets()
        self.update_values()
        
    def create_widgets(self):
        # 状态变量显示
        self.lbl_serial = ttk.Label(self, text="串口数据包计数:")
        self.lbl_serial_val = ttk.Label(self, text="0")
        
        self.lbl_arrive = ttk.Label(self, text="到达状态:")
        self.lbl_arrive_val = ttk.Label(self, text="0")
        
        # 布局
        self.lbl_serial.grid(row=0, column=0, padx=5, pady=5)
        self.lbl_serial_val.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_arrive.grid(row=1, column=0, padx=5, pady=5)
        self.lbl_arrive_val.grid(row=1, column=1, padx=5, pady=5)
        
    def update_values(self):
        """定时更新界面数值"""
        # 从订阅者节点获取最新值
        self.lbl_serial_val.config(text=str(self.sub_node.serial.data_num))
        self.lbl_arrive_val.config(text=str(self.sub_node.serial.ifArrive_int))
        self
        
        # 每200ms刷新一次
        self.after(200, self.update_values)
