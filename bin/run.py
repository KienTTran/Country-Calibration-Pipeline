import time
import math

def main():
    start_time = time.time()
    duration = 120  # seconds

    # Allocate ~1GB RAM: 1024 blocks of 1MB
    memory_hog = ["A" * 1024 * 1024 for _ in range(1024)]

    # Full CPU load loop
    while time.time() - start_time < duration:
        x = 0.0
        for i in range(100000):
            x += math.sqrt(i) * math.sin(i)  # CPU-intensive floating point ops

if __name__ == "__main__":
    main()
