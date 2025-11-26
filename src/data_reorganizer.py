import webdataset as wds
import os
from tqdm import tqdm
import sys
import glob

# ================= 配置区域 =================

# 1. 输入路径：支持 glob 通配符
# 例如你的多线程代码生成了一堆散碎文件：part-000000.tar, part-000001.tar ...
# SOURCE_PATTERN = glob.glob("ih26m_train_wds_output/ih26m_train-worker*.tar")
SOURCE_PATTERN = glob.glob(os.environ["SRC"])

# 2. 输出路径：最终合并后的文件命名格式
# %06d 会自动递增：ih26m_val-000000.tar, ih26m_val-000001.tar ...
# OUTPUT_PATTERN = "/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train2/%06d.tar"
OUTPUT_PATTERN = os.environ["DST"]

# 3. 限制参数
# 3GB (推荐值：机械硬盘 1GB-3GB，SSD 200MB-500MB)
MAX_SIZE = 3 * 1024 * 1024 * 1024

# 单个文件最大样本数 (设得很大，让 maxsize 起主导作用)
MAX_COUNT = 1000000

# ===========================================

def repack_without_decoding():
    # 0. 检查输出目录
    output_dir = os.path.dirname(OUTPUT_PATTERN)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"源数据: {SOURCE_PATTERN}")
    print(f"目标数据: {OUTPUT_PATTERN}")
    print(f"单个分片限制: {MAX_SIZE / (1024**3):.2f} GB")

    # 1. 创建读取器 (Reader)
    # 关键点：不调用 .decode()
    # WebDataset 会直接读取 tar 中的文件内容为 bytes
    # sample 格式示例: {'__key__': '0001', 'jpg': b'\xff\xd8...', 'json': b'{...}'}
    dataset = wds.WebDataset(SOURCE_PATTERN)

    # 2. 创建写入器 (Writer)
    # ShardWriter 会自动检测输入字典的 key 后缀 (如 .jpg, .npy) 并写入 tar
    sink = wds.ShardWriter(OUTPUT_PATTERN, maxsize=MAX_SIZE, maxcount=MAX_COUNT)

    count = 0
    total_bytes = 0

    try:
        # 使用 tqdm 显示进度 (由于是流式读取，不知道总数，只显示当前处理数)
        pbar = tqdm(dataset, desc="Merging Shards", unit="samples")

        for sample in pbar:
            # 直接透传 (Pass-through)
            sink.write(sample)

            count += 1

            # 简单的统计信息更新 (可选)
            # 估算写入量：累加所有 value 的长度
            sample_bytes = sum(len(v) for k, v in sample.items() if isinstance(v, bytes))
            total_bytes += sample_bytes

            if count % 100 == 0:
                pbar.set_postfix({"Data": f"{total_bytes / (1024**3):.2f} GB"})

    except Exception as e:
        print(f"\n[Error] 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 3. 确保关闭文件流，完成最后一个 tar 的写入
        sink.close()
        print(f"\n合并完成！共处理 {count} 个样本。")
        print(f"数据已保存至: {output_dir}")

if __name__ == "__main__":
    # 简单的防止误操作检查
    if SOURCE_PATTERN == OUTPUT_PATTERN:
        print("错误：输入路径和输出路径模式不能相同，这会覆盖源文件！")
        sys.exit(1)

    repack_without_decoding()