import os
from datetime import datetime
from jinja2 import Template

def generate_report(opencv_stats, torch_stats, image_count):
    """生成完整的实验报告"""
    # 准备报告数据
    report_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'opencv': {
            'mean_time': f"{opencv_stats[0]:.4f}",
            'std_dev': f"{opencv_stats[1]:.4f}",
            'throughput': f"{opencv_stats[2]:.1f}"
        },
        'torch': {
            'mean_time': f"{torch_stats[0]:.4f}",
            'std_dev': f"{torch_stats[1]:.4f}",
            'throughput': f"{torch_stats[2]:.1f}"
        },
        'speedup': f"{torch_stats[0]/opencv_stats[0]:.2f}",
        'image_count': image_count,
        'performance_chart': 'performance_comparison.png',
        'opencv_samples': sorted(os.listdir('opencv_results')),
        'torch_samples': sorted(os.listdir('torchvision_results'))
    }

    # 报告模板
    template = Template('''
# 图像增强性能实验报告

**实验日期**: {{ date }}  
**测试图像数量**: {{ image_count }}

## 性能对比结果

| 指标 | OpenCV | Torchvision |
|------|--------|-------------|
| 平均处理时间(s) | {{ opencv.mean_time }} ± {{ opencv.std_dev }} | {{ torch.mean_time }} ± {{ torch.std_dev }} |
| 吞吐量(imgs/s) | {{ opencv.throughput }} | {{ torch.throughput }} |

**加速比**: {{ speedup }}x

![性能对比图]({{ performance_chart }})

## 增强效果示例

### OpenCV 增强结果
{% for img in opencv_samples %}
![OpenCV增强示例](opencv_results/{{ img }})
{% endfor %}

### Torchvision 增强结果
{% for img in torch_samples %}
![Torchvision增强示例](torchvision_results/{{ img }})
{% endfor %}
''')

    # 生成报告
    with open('experiment_report.md', 'w', encoding='utf-8') as f:
        f.write(template.render(**report_data))
    
    print("\n实验报告已生成: experiment_report.md")
