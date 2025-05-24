import time
import cv2
import numpy as np
from PIL import Image
import glob
import torch
import torchvision.transforms as torch_transforms
from opencv_transforms import transforms as opencv_transforms
from report_generator import generate_report

def build_opencv_pipeline():
    """构建基于OpenCV的图像增强管道"""
    return opencv_transforms.Compose([
        opencv_transforms.Resize(256),  # 将图像调整大小到256x256像素
        opencv_transforms.RandomRotation(degrees=10),  # 随机旋转图像，旋转角度范围为±10度
        opencv_transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图像，概率为0.5
        opencv_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        opencv_transforms.ToTensor(),  # 将图像转换为PyTorch张量
    ])

def build_torchvision_pipeline():
    """构建基于torchvision的图像增强管道"""
    return torch_transforms.Compose([
        torch_transforms.Resize(256),  # 将图像调整大小到256x256像素
        torch_transforms.RandomRotation(degrees=10),  # 随机旋转图像，旋转角度范围为±10度
        torch_transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图像，概率为0.5
        torch_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整图像的亮度、对比度、饱和度和色调
        torch_transforms.ToTensor(),  # 将图像转换为PyTorch张量
    ])

def apply_pipeline_to_image(pipeline, img_file):
    """应用管道到单个图像并返回增强结果"""
    if isinstance(pipeline, torch_transforms.Compose): # 如果是基于torchvision的管道
        img = Image.open(img_file)
        augmented = pipeline(img)
        if isinstance(augmented, torch.Tensor):
            augmented = augmented.numpy().transpose(1, 2, 0)
    else: # 如果是基于OpenCV的管道
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = pipeline(img)
        if isinstance(augmented, torch.Tensor):
            augmented = augmented.numpy().transpose(1, 2, 0)
    return img, augmented

def save_augmented_images(original_img, augmented_img, output_dir="augmented_results", index=0):
    """保存原始和增强后的图像对比"""
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img if isinstance(original_img, np.ndarray) else np.array(original_img))
    plt.title("原始图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_img)
    plt.title("增强后图像")
    plt.axis('off')
    
    plt.savefig(f"{output_dir}/result_{index}.png")
    plt.close()

from tqdm import tqdm

def test_pipeline_performance(pipeline, image_files, num_runs=10):
    """测试给定管道的性能"""
    pipeline_type = "OpenCV" if isinstance(pipeline, opencv_transforms.Compose) else "Torchvision"
    print(f"\n=== 开始测试 {pipeline_type} 增强管道性能 ===")
    print(f"测试配置: 图像数量={len(image_files)}, 运行次数={num_runs}")
    # 预加载所有图像到内存
    loaded_images = []
    for img_file in tqdm(image_files, desc="预加载图像"):
        if isinstance(pipeline, torch_transforms.Compose):
            img = Image.open(img_file)
        else:
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        loaded_images.append(img)
    
    times = []  # 存储执行时间
    
    # 预热(避免首次运行的开销影响结果)
    for img in loaded_images[:1]:
        _ = pipeline(img)
    
    # 性能测试运行
    for _ in tqdm(range(num_runs), desc="性能测试进度"):
        for img in loaded_images:
            # 只测量增强处理时间
            start = time.perf_counter()
            _ = pipeline(img)
            end = time.perf_counter()
            times.append(end - start)
    
    # 性能测试结束后计算统计指标
    total_time = sum(times)
    total_images = len(loaded_images) * num_runs
    per_image_time = total_time / total_images
    std_per_image = np.std(times)  # 直接使用原始时间标准差
    imgs_per_sec = 1.0 / per_image_time
    
    return per_image_time, std_per_image, imgs_per_sec

def main():
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 设置随机种子确保可重复性
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 获取测试图像
    image_files = glob.glob('test_images/*.jpg')
    
    # 构建管道
    opencv_pipeline = build_opencv_pipeline()
    torch_pipeline = build_torchvision_pipeline()
    
    # 测试性能
    opencv_mean, opencv_std, opencv_ips = test_pipeline_performance(opencv_pipeline, image_files)
    torch_mean, torch_std, torch_ips = test_pipeline_performance(torch_pipeline, image_files)
    
    # 打印结果
    from tabulate import tabulate
    
    table_data = [
        ["Avg time per image(s)", f"{opencv_mean:.4f} ± {opencv_std:.4f}", f"{torch_mean:.4f} ± {torch_std:.4f}"],
        ["Throughput(imgs/s)", f"{opencv_ips:.1f}", f"{torch_ips:.1f}"]
    ]
    
    print("\nPerformance Comparison:")
    print(tabulate(table_data, 
                  headers=["Metric", "OpenCV", "Torchvision"],
                  tablefmt="grid"))
    print(f"\n加速比: {torch_mean/opencv_mean:.2f}x")
    
    # 绘制性能对比图
    plt.figure(figsize=(10, 5))
    bars = plt.bar(['OpenCV', 'Torchvision'], [opencv_ips, torch_ips])
    plt.ylabel('吞吐量 (imgs/s)')
    plt.title('图像增强性能对比')
    
    # 添加数值标签和y轴刻度
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.yticks(plt.yticks()[0], rotation=45)
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()
    
    # 输出CSV格式的性能摘要
    with open('performance_summary.csv', 'w') as f:
        f.write("method,per_image_s,std_s,imgs_per_s\n")
        f.write(f"OpenCV,{opencv_mean:.4f},{opencv_std:.4f},{opencv_ips:.1f}\n")
        f.write(f"Torchvision,{torch_mean:.4f},{torch_std:.4f},{torch_ips:.1f}\n")
    
    # 处理并保存所有示例图像
    if image_files:
        print("\n生成增强结果可视化...")
        # 重置随机种子确保可视化一致性
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # OpenCV管道
        for i, img_file in enumerate(tqdm(image_files, desc="OpenCV增强进度")):
            original_img, opencv_augmented = apply_pipeline_to_image(opencv_pipeline, img_file)
            save_augmented_images(original_img, opencv_augmented, "opencv_results", i)
        
        # Torchvision管道
        for i, img_file in enumerate(tqdm(image_files, desc="Torchvision增强进度")):
            original_img, torch_augmented = apply_pipeline_to_image(torch_pipeline, img_file)
            save_augmented_images(original_img, torch_augmented, "torchvision_results", i)

if __name__ == "__main__":
    main()
    # 生成实验报告
    image_files = glob.glob('test_images/*.jpg')
    opencv_stats = test_pipeline_performance(build_opencv_pipeline(), image_files)
    torch_stats = test_pipeline_performance(build_torchvision_pipeline(), image_files)
    generate_report(opencv_stats, torch_stats, len(image_files))
