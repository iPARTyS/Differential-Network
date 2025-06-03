import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
# 创建示例数据
x = np.array(['UNet', 'TransUNet', "PVT-CASCADE", 'Ours'])
y1 = np.array([70.11, 77.61, 81.06, 83.33],)  # dice
y2 = np.array([34.52, 105.28, 35.28, 24.13])  # parameter
y3 = np.array([59.39, 67.32, 70.88, 74.68])  # miou
y4 = np.array([13.1, 25.39, 6.26, 70.0])  # flops
# 创建第一个y轴的柱状图

fig, ax1 = plt.subplots()

# color = (0.16, 0.52, 0.47)  # (R, G, B) 元组
# color = (0.25, 0.64, 0.6)  # (R, G, B) 元组
color = (0.30, 0.13, 0.53)
ax1.set_xlabel('Models')
ax1.set_ylabel('Model Parameters And Computational Complexity', color='black')
bars = ax1.bar(x, y2, color=color, width=0.2, alpha=0.7, label='Params(M)')
ax1.tick_params(axis='y', labelcolor='black')
# 设置第一个y轴刻度范围
ax1.set_ylim(0, 120)
#
# # 调整第一个y轴刻度
ax1.yaxis.set_ticks(np.arange(0, 120, 10))
bars_iou = ax1.bar(np.arange(len(x)) + 0.3, y4, color=(0.24, 0.51, 0.49), width=0.2, label='Flops(G)', align='center')
# 在柱状图上添加数值标签
for bar, val in zip(bars, y2):
    ax1.text(bar.get_x() + bar.get_width() / 2, val, round(val, 2), ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars_iou, y4):
     ax1.text(bar.get_x() + bar.get_width() / 2, val, round(val, 2), ha='center', va='bottom', fontsize=9)
# 创建第二个y轴的折线图
ax2 = ax1.twinx()
# 使用RGB值来定义颜色（蓝色）
# color = (0.94, 0.63, 0.12)  # (R, G, B) 元组
# color = (0.43, 0.75, 0.91)
color = (0.87, 0.70, 0.63)
# 调整第二个y轴标签的位置，设置相对坐标
ax2.yaxis.set_label_coords(1.09, 0.5)
ax2.set_ylabel('Average Dice (%)', color='black')
ax2.plot(x, y1, color=color, linestyle='-', marker='o', markersize=10, label='Dice (%)')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(50, 87)

# 调整第二个y轴刻度
ax2.yaxis.set_ticks(np.arange(50, 87, 4))
# 在折线图上添加数值标签
for x_val, y_val in zip(x, y1):
    ax2.text(x_val, y_val+1.0, round(y_val, 2), ha='center', va='bottom', fontsize=9)

# 添加第二个y轴的折线图

# 在折线图上添加数值标签

# 隐藏刻度线
ax1.tick_params(axis='both', which='both', length=0)
ax2.tick_params(axis='y', which='both', length=0)
# plt.subplots_adjust(right=0.05)
# 显示图例
# 设置图形的布局
plt.subplots_adjust(right=0.85)
# 设置x轴刻度位置和标签
ax1.set_xticks(np.arange(len(x)) + 0.15)
ax1.set_xticklabels(x)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
all_handles = lines + lines2
all_labels = labels + labels2
# legend1 = ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True)
# legend1.get_frame().set_alpha(1)
# legend2 = ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True)
# legend2.get_frame().set_alpha(1)

legend = fig.legend(all_handles, all_labels, loc='center', ncol=3, frameon=True, framealpha=1, handletextpad=0.5,
                    handlelength=2, shadow=True)
legend.set_bbox_to_anchor((0.45, 0.95))
legend.get_frame().set_alpha(1)

# 保存图形
plt.savefig('your_plot.png', bbox_inches='tight')

plt.show()