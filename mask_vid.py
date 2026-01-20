import imageio
import sys
import os
import numpy as np

def mask_video(input_path, x, y, dx, dy):
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    # 初始化读取器和写入器
    # 使用 imageio.get_reader 可以方便地获取视频元数据
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    fps = meta.get('fps', 24)
    
    # 准备输出文件，明确指定使用 libx264 编码器
    # quality 参数控制压缩质量（0-10，默认 5），pixelformat 确保浏览器兼容性
    output_path = "output_masked.mp4"
    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        codec='libx264', 
        pixelformat='yuv420p', 
        quality=8
    )

    print(f"正在处理视频，编码格式：libx264...")

    for frame in reader:
        # imageio 读取的 frame 是只读的 numpy 数组，需创建副本或直接操作
        # 注意：imageio 默认读取为 RGB 格式
        height, width, _ = frame.shape
        
        # 计算切片边界
        y_start, y_end = max(0, y), min(y + dy, height)
        x_start, x_end = max(0, x), min(x + dx, width)
        
        # 核心遮罩逻辑
        # 此时的 frame 已经是 numpy 数组，直接修改指定区域
        frame[y_start:y_end, x_start:x_end] = [127, 127, 127]
        
        writer.append_data(frame)

    reader.close()
    writer.close()
    print(f"处理完成。输出文件：{output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("用法: python mask_video.py input_vid.mp4 x y dx dy")
    else:
        try:
            mask_video(
                sys.argv[1], 
                int(sys.argv[2]), 
                int(sys.argv[3]), 
                int(sys.argv[4]), 
                int(sys.argv[5])
            )
        except Exception as e:
            print(f"发生错误: {e}")