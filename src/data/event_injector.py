import numpy as np

class ECommerceEventInjector:
    def __init__(self, timestamps, req_rate_array):
        self.timestamps = timestamps
        self.req_rate = req_rate_array
        self.num_samples = len(req_rate_array)

    def inject_mega_sales(self, target_months=[11, 12], target_days=[11, 12], multiplier=8.0):
        print(f"[*] Injecting Mega Sale events for {target_days[0]}/{target_months[0]}...")
        for i, ts in enumerate(self.timestamps):
            if ts.month in target_months and ts.day in target_days:
                if 0 <= ts.hour < 1:
                    self.req_rate[i] *= (multiplier * np.random.uniform(0.8, 1.2))
                elif 12 <= ts.hour < 13:
                    self.req_rate[i] *= (multiplier * 0.6 * np.random.uniform(0.8, 1.2))
                elif 20 <= ts.hour < 22:
                    self.req_rate[i] *= (multiplier * 0.8 * np.random.uniform(0.8, 1.2))
        return self.req_rate

    def inject_payday_sales(self, multiplier=3.0):
        print("[*] Injecting Payday Sales events...")
        for i, ts in enumerate(self.timestamps):
            if ts.day in [15, 25]:
                if 19 <= ts.hour <= 23:
                    self.req_rate[i] *= (multiplier * np.random.uniform(0.8, 1.2))
        return self.req_rate
