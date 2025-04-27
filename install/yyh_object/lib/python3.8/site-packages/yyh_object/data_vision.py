import matplotlib.pyplot as plt
import numpy as np

class ImageVisualizer:
    def __init__(self, variables=None):
        self.variables = variables if variables is not None else {}
        self.fig, self.ax = plt.subplots(figsize=(12, 6))  # 调整窗口大小
        self.ax.set_title('Variables Show')
        plt.ion()  # 交互模式
        plt.show()
        
        # 添加网格线
        self.ax.grid(True)
        
        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 连接拖动事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.zoom_state = False

    def update_variable(self, var_name, var_value):
        if var_name not in self.variables:
            self.variables[var_name] = []
        self.variables[var_name].append(var_value)
        self.ax.cla()  # 清除当前轴
        for name, values in self.variables.items():
            self.ax.plot(values, label=name)
        self.ax.legend()
        self.ax.set_title('Variables')
        self.ax.grid(True)  # 保持网格线显示
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_click(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            print(f"Clicked at x={x}, y={y}")

    def on_key_press(self, event):
        if event.key == 'z':
            if not self.zoom_state:
                self.zoom_state = True
                self.ax.set_xlim(event.xdata - 1, event.xdata + 1)
                self.ax.set_ylim(event.ydata - 1, event.ydata + 1)
            else:
                self.zoom_state = False
                self.ax.autoscale()
            self.fig.canvas.draw_idle()

# # 示例使用
# if __name__ == "__main__":
#     # 假设我们有一些变量数据
#     visualizer = ImageVisualizer()

#     # 模拟更新变量
#     for i in range(100):
#         # 更新变量数据
#         visualizer.update_variable('var1', np.sin(i * 0.1))
#         visualizer.update_variable('var2', np.cos(i * 0.1))
#         visualizer.update_variable('var3', np.tan(i * 0.1))