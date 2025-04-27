import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons

class ImageVisualizer:
    def __init__(self, variables=None):
        self.variables = variables if variables is not None else {}
        self.fig = plt.figure(figsize=(14, 7))
        
        # 主图表区域 (留出空间给复选框)
        self.ax = plt.axes([0.1, 0.2, 0.7, 0.7])
        self.ax.set_title('Interactive Variables Visualization')
        plt.ion()
        
        # 初始化图形元素
        self.ax.grid(True)
        self.annotation = None
        self.lines = {}  # 存储各变量的线对象
        self.visible = {}  # 存储各变量的可见状态
        
        # 复选框区域
        self.check_ax = plt.axes([0.82, 0.3, 0.15, 0.4])
        self.check_ax.set_title('Variables')
        self.check_ax.axis('off')
        self.checkboxes = None
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 视图状态
        self.zoom_state = False
        self.current_xlim = None
        self.current_ylim = None
        
        plt.show()

    def update_variable(self, var_name, var_value):
        """更新变量数据并刷新视图"""
        if var_name not in self.variables:
            self.variables[var_name] = []
            self.visible[var_name] = True  # 默认可见
        
        self.variables[var_name].append(var_value)
        
        # 保存当前视图状态
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()
        
        # 重绘图表
        self.redraw_plot()
        
        # 更新复选框
        self.update_checkboxes()
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def redraw_plot(self):
        """重绘所有可见的曲线"""
        self.ax.cla()
        self.lines = {}
        
        for name, values in self.variables.items():
            if self.visible.get(name, True):
                line, = self.ax.plot(values, label=name, marker='o', markersize=4)
                self.lines[name] = line
        
        # 恢复视图状态
        if self.zoom_state and self.current_xlim is not None:
            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)
        else:
            self.ax.autoscale_view()
        
        self.ax.legend(loc='upper left')
        self.ax.grid(True)
        self.ax.set_title('Interactive Variables Visualization')

    def update_checkboxes(self):
        """更新复选框控件"""
        if self.checkboxes:
            self.check_ax.cla()
            self.check_ax.axis('off')
        
        labels = list(self.variables.keys())
        actives = [self.visible.get(name, True) for name in labels]
        
        self.checkboxes = CheckButtons(
            ax=self.check_ax,
            labels=labels,
            actives=actives
        )
        
        # 设置复选框样式
        for rect, label in zip(self.checkboxes.rectangles, self.checkboxes.labels):
            rect.set_facecolor('lightblue')
            rect.set_edgecolor('gray')
            label.set_fontsize(10)
        
        self.checkboxes.on_clicked(self.toggle_variable)

    def toggle_variable(self, label):
        """切换变量的可见状态"""
        self.visible[label] = not self.visible.get(label, True)
        self.redraw_plot()
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """点击事件处理：显示数据点信息"""
        if event.inaxes != self.ax:
            return
        
        # 清除旧注释
        if self.annotation:
            self.annotation.remove()
        
        # 计算最近的数据索引
        x_index = int(round(event.xdata))
        annotations = []
        
        # 收集所有可见变量在该索引处的值
        for name, values in self.variables.items():
            if self.visible.get(name, True) and x_index < len(values):
                y_val = values[x_index]
                annotations.append(f"{name} [{x_index}]: {y_val:.4f}")
        
        if annotations:
            # 创建新注释
            text = "\n".join(annotations)
            self.annotation = self.ax.annotate(
                text,
                xy=(x_index, event.ydata),
                xytext=(20, -40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='gray'),
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9)
            )
            self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        """按键处理：z键缩放功能"""
        if event.key == 'z' and event.inaxes == self.ax:
            current_x = event.xdata
            current_y = event.ydata
            
            if not self.zoom_state:  # 进入缩放模式
                zoom_factor = 0.5  # 缩放因子
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                
                new_x_width = (xlim[1] - xlim[0]) * zoom_factor
                new_y_height = (ylim[1] - ylim[0]) * zoom_factor
                
                self.ax.set_xlim(current_x - new_x_width/2, current_x + new_x_width/2)
                self.ax.set_ylim(current_y - new_y_height/2, current_y + new_y_height/2)
                self.zoom_state = True
            else:  # 退出缩放模式
                self.ax.autoscale_view()
                self.zoom_state = False
            
            self.fig.canvas.draw_idle()

# 示例使用
if __name__ == "__main__":
    viz = EnhancedVisualizerWithCheckboxes()
    
    # 模拟实时数据更新
    for t in range(100):
        x = t * 0.1
        viz.update_variable('Sine', np.sin(x))
        viz.update_variable('Cosine', np.cos(x))
        viz.update_variable('Exponential', np.exp(-x/2))
        viz.update_variable('Linear', x/5)
        plt.pause(0.05)  # 控制更新频率