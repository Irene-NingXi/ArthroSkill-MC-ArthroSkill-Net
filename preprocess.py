"""
数据预处理入口脚本
用法: python preprocess.py --input ./videos --output ./processed
"""
import argparse
from utils import VideoPreprocessor


def main():
    parser = argparse.ArgumentParser(description="预处理关节镜手术视频")
    parser.add_argument("--input", "-i", required=True, help="原始视频目录")
    parser.add_argument("--output", "-o", default="./processed", help="输出目录")
    parser.add_argument("--clip-len", type=int, default=16, help="片段长度")
    parser.add_argument("--fps", type=int, default=25, help="采样帧率")
    parser.add_argument("--brightness", type=int, default=30, help="亮度阈值")
    parser.add_argument("--blur", type=int, default=100, help="模糊度阈值")
    args = parser.parse_args()

    preprocessor = VideoPreprocessor(
        clip_len=args.clip_len,
        fps=args.fps,
        brightness_thresh=args.brightness,
        blur_thresh=args.blur
    )

    total = preprocessor.process_directory(args.input, args.output)
    print(f"\n预处理完成，共生成 {total} 个片段")


if __name__ == "__main__":
    main()
