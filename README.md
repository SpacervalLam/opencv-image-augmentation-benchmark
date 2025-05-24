# 基于 OpenCV 的高效图像增强实验

## 背景简介

图像增强是计算机视觉任务中的重要步骤，用于扩展数据集、多样化输入数据，进而提高模型的泛化能力。Torchvision 提供了常用的图像增强变换，但由于依赖于 Pillow，其速度可能成为瓶颈。[`opencv_transforms`](https://github.com/jbohnslav/opencv_transforms) 项目基于 OpenCV 实现了 Torchvision 的图像增强功能，显著提升了处理速度。本次实验将熟悉 OpenCV 的图像增强方法，并探索其性能优势。

---

## 实验任务

### 任务 1：项目环境搭建

1. **克隆仓库**  
   克隆 `opencv_transforms` 项目到本地开发环境，确保代码能够正常运行，测试运行是否有报错。

### 任务 2：图像增强流水线

1. **构建流水线**  
   使用 `opencv_transforms` 构建图像增强流水线，包括以下操作：
   - `Resize`
   - `RandomRotation`
   - `RandomHorizontalFlip`
   - `ColorJitter`

2. **输入输出**  
   输入一组测试图像，输出增强后的图像。

3. **性能对比**  
   测试流水线的运行时间，并与 Torchvision 实现的相同流水线进行对比。

### 任务 3：扩展与优化

1. **大规模数据集测试**  
   - 使用 Cityscapes 数据集（或其他批量图像数据集）测试 OpenCV 和 Torchvision 的增强性能。
   - 比较两种实现的批量图像处理速度。

2. **自定义增强操作**  
   - 在 `opencv_transforms` 中实现一个自定义增强操作（如 `RandomCrop` 或 `GaussianBlur`）。
   - 测试自定义操作的性能，并与 Torchvision 中的对应操作进行对比。

3. **多线程优化**  
   - 探索 OpenCV 在多线程环境下的图像增强性能（如在 `num_workers > 0` 的 PyTorch DataLoader 中使用）。
   - 分析多线程对增强速度的影响。

4. **高分辨率图像增强**  
   - 使用高分辨率图像（如 4K 图像）测试 OpenCV 和 Torchvision 的增强性能。
   - 比较两种实现的处理速度和内存占用情况。

5. **深入性能分析**  
   - 使用性能分析工具（如 `cProfile` 或 `timeit`）深入分析 OpenCV 和 Torchvision 的性能差异。

### 任务 4：实战尝试

1. **自定义增强函数**  
   实现基于 OpenCV 的至少四种随机图像增强操作，并创建一个自定义的 PyTorch Dataset 类加载 CIFAR-10 数据并应用这些增强操作。

2. **对比实验**  
   使用标准 `torchvision.transforms` 实现类似增强的数据加载器，训练同一模型结构，比较两者的性能。

---

## 实验结果

1. **代码文件**  
   - 任务 2：图像增强流水线 [transforms_pipeline.py](transforms_pipeline.py)

2. **实验结果**  
   - 提交测试运行日志和性能对比结果，以表格或图表形式展示。
   - 可视化结果图片整理到 `results/` 文件夹中。
   - 

3. **实验报告**  
   - [实验报告](experiment_report.md)

---

## 附加说明

- **参考资料**：
  - [OpenCV 官方文档](https://docs.opencv.org/)
  - [Torchvision 官方文档](https://pytorch.org/vision/stable/index.html)
  - [opencv_transforms 代码仓库](https://github.com/jbohnslav/opencv_transforms)