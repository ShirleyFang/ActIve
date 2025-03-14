import random

class WearableSimulator:
    def __init__(self):
        """ 初始化模拟的可穿戴设备数据 """
        self.sedentary_hours = random.uniform(6, 12)  # 假设久坐 6~12 小时
        self.avg_heart_rate = random.randint(60, 80)  # 正常心率
        self.peak_heart_rate = random.randint(100, 130)  # 运动或压力下的心率峰值

    def get_data(self):
        """ 获取模拟的可穿戴设备数据 """
        return {
            "sedentary_hours": round(self.sedentary_hours, 1),
            "avg_heart_rate": self.avg_heart_rate,
            "peak_heart_rate": self.peak_heart_rate
        }

# 测试代码
if __name__ == "__main__":
    wearable = WearableSimulator()
    print(wearable.get_data())
