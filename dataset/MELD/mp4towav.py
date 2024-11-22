import os
from ffmpy3 import FFmpeg


def solve(dataset):
    for key, value in dataset.items():
        filepath = value[0]  # 存放mp4视频的path
        output_wav_dir = value[1]  #输出wav的path
        if not os.path.exists(output_wav_dir):
            os.mkdir(output_wav_dir)
        for file in os.listdir(filepath):
            if file.startswith('dia') and file.endswith('.mp4'):
                input_file = os.path.join(filepath, file)
                output_file = os.path.join(output_wav_dir, file.replace('.mp4', '.wav'))
                if os.path.exists(output_file):
                    continue
                ff = FFmpeg(
                    inputs={input_file: None},
                    outputs={output_file: None},
                )
                ff.run()


if __name__ == '__main__':
    train_path = "/home/wanghanyang/Public/DATASET/MELD/train_splits"  #放mp4的文件夹
    train_outwav_path = "/home/wanghanyang/Public/DATASET/MELD/train_splits_wav"  #输出wav的文件夹
    dev_path = "/home/wanghanyang/Public/DATASET/MELD/dev_splits_complete"
    dev_outwav_path = "/home/wanghanyang/Public/DATASET/MELD/dev_splits_wav"
    test_path = "/home/wanghanyang/Public/DATASET/MELD/output_repeated_splits_test"
    test_outwav_path = "/home/wanghanyang/Public/DATASET/MELD/test_splits_wav"
    dataset = {
        'train': (train_path, train_outwav_path),
        'dev': (dev_path, dev_outwav_path),
        'test': (test_path, test_outwav_path)
    }
    solve(dataset)