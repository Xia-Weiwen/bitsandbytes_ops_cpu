# bitsandbytes_ops_cpu

Python implementation of bisandbytes ops for CPU

**Run tests**

```bash
python test_ops.py
```

**Run benchmarks**

Eager vs. inductor (torch.compile)
```bash
# Latency
numactl -m <node id> -C <core list> python benchmarks.py --all
# Profiling
numactl -m <node id> -C <core list> python benchmarks.py --all --profile
```
