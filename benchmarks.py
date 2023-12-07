import argparse
import torch
import sys
sys.path.append(sys.path[0])
from int8_ops import (
    double_quant_eager,
    double_quant,
    transform_eager,
    transform,
    igemmlt,
    mm_dequant_eager,
    mm_dequant,
    extract_outliers_eager,
    extract_outliers,
)
import time

parser = argparse.ArgumentParser(description="Benchmarks for bnb int8 ops, eager vs inductor")
parser.add_argument("--num-active", default=20, type=int, help="number of active iterations for benchmark")
parser.add_argument("--num-warmup", default=10, type=int, help="number of warmup iterations for benchmark")
parser.add_argument("--all", action="store_true", help="Run all benchmarks")
parser.add_argument("--double-quant", action="store_true", help="Run benchmark for the double_quant op")
parser.add_argument("--transform", action="store_true", help="Run benchmark for the transform op")
parser.add_argument("--mm-dequant", action="store_true", help="Run benchmark for the mm_dequant op")
parser.add_argument("--extract-outliers", action="store_true", help="Run benchmark for the extract_outliers op")
parser.add_argument("--profile", action="store_true", help="Run all benchmarks with PyTorch profiler")
args = parser.parse_args()


def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=-1))


def run_benchmark(func_name, func_eager, func_inductor, *func_args, **func_kwargs):
    with torch.no_grad():
      # Eager
      for _ in range(args.num_warmup):
        func_eager(*func_args, **func_kwargs)
      t0 = time.time()
      for _ in range(args.num_active):
        func_eager(*func_args, **func_kwargs)
      latency_eager = (time.time() - t0) / args.num_active
      latency_eager = round(latency_eager * 1000, 3)
      # Inductor
      for _ in range(args.num_warmup):
        func_inductor(*func_args, **func_kwargs)
      t0 = time.time()
      for _ in range(args.num_active):
        func_inductor(*func_args, **func_kwargs)
    latency_inductor = (time.time() - t0) / args.num_active
    latency_inductor = round(latency_inductor * 1000, 3)
    print(f"--- Benchmkark for {func_name} ---")
    if args.profile:
      print("\nProfiling for eager")
      with torch.no_grad(), torch.profiler.profile(
          activities=[torch.profiler.ProfilerActivity.CPU],
          schedule=torch.profiler.schedule(
              wait=0, warmup=3, active=1, repeat=0),
          on_trace_ready=trace_handler
          ) as p:
              for _ in range(4):
                  func_eager(*func_args, **func_kwargs)
                  p.step()
      print("\nProfiling for inductor")
      with torch.no_grad(), torch.profiler.profile(
          activities=[torch.profiler.ProfilerActivity.CPU],
          schedule=torch.profiler.schedule(
              wait=0, warmup=3, active=1, repeat=0),
          on_trace_ready=trace_handler
          ) as p:
              for _ in range(4):
                  func_inductor(*func_args, **func_kwargs)
                  p.step()
    print("\n--- Summary ---")
    print(f"Eager latency: {latency_eager} ms, inductor latency: {latency_inductor} ms, "
          f"speedup: {round(latency_eager - latency_inductor, 3)} ms ({round(latency_eager / latency_inductor - 1, 4) * 100}%)")
    print("----------\n")


if args.double_quant or args.all:
    A = torch.rand(4096, 4096) * 3.0
    threshold = 3.0
    func_args = (A,)
    func_kwargs = {'threshold': threshold}
    run_benchmark('double_quant', double_quant_eager, double_quant, *func_args, **func_kwargs)

if args.transform or args.all:
    A = torch.rand(4096, 4096)
    func_args = (A,)
    func_kwargs = {'transpose': True}
    run_benchmark('transform', transform_eager, transform, *func_args, **func_kwargs)

if args.mm_dequant or args.all:
    shapeA, shapeB = (4096, 4096), (4096, 4096)
    A = torch.rand(shapeA)
    A_min, A_max = A.aminmax(dim=-1)
    A_stats = torch.max(A_max, A_min.neg())
    A_scale = A_stats / 127
    A_int8 = torch.round(A / A_scale.unsqueeze(-1)).to(torch.int8)
    B = torch.randn(shapeB)
    B_min, B_max = B.aminmax(dim=-1)
    B_stats = torch.max(B_max, B_min.neg())
    B_scale = B_stats / 127
    B_int8 = torch.round(B / B_scale.unsqueeze(-1)).to(torch.int8)
    bias = torch.randn(shapeB[0])
    C_i32, _ = igemmlt(A_int8, B_int8)
    func_args = (C_i32, None, A_stats, B_stats)
    func_kwargs = {'bias': bias}
    run_benchmark('mm_dequant', mm_dequant_eager, mm_dequant, *func_args, **func_kwargs)

if args.extract_outliers or args.all:
    shapeA = (4096, 4096 * 4)
    idx = torch.unique(torch.randint(0, shapeA[1], size=(10,)).int())
    A = torch.randint(-128, 127, size=shapeA).to(torch.int8)
    func_args = (A, None, idx)
    func_kwargs = {}
    run_benchmark('extract_outliers', extract_outliers_eager, extract_outliers, *func_args, **func_kwargs)
