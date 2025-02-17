import time
import subprocess

def get_gpu_usage(pid):
    # 执行 nvidia-smi 命令获取 GPU 使用情况
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split('\n')
    for line in lines:
        columns = line.split(', ')
        if columns[0] == str(pid):
            gpu_util = int(columns[1].replace('%', ''))
            mem_used = int(columns[2].replace(' MiB', ''))
            mem_total = int(columns[3].replace(' MiB', ''))
            return gpu_util, mem_used, mem_total
    return None, None, None

def monitor_gpu_usage(pid, duration=60, interval=1):
    max_gpu_util = 0
    max_mem_used = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        gpu_util, mem_used, mem_total = get_gpu_usage(pid)
        if gpu_util is not None:
            max_gpu_util = max(max_gpu_util, gpu_util)
            max_mem_used = max(max_mem_used, mem_used)
        time.sleep(interval)
    return max_gpu_util, max_mem_used

if __name__ == '__main__':
    pid = 4110055  # 替换为您的目标进程 PID
    duration = 360  # 监控持续时间，单位：秒
    interval = 1  # 采样间隔，单位：秒
    max_gpu_util, max_mem_used = monitor_gpu_usage(pid, duration, interval)
    print(f"最大 GPU 利用率: {max_gpu_util}%")
    print(f"最大显存占用: {max_mem_used} MiB")
