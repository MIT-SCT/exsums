unsigned int cpu_workload(void) {
  unsigned int x = 0;
  for (size_t ind = 0; ind < 6174; ind++) {
    x = ((x ^ 0x123) + x * 3) % 123456;
  }
  return x;
}

[[clang::optnone]]
long fib(long n) {
  if (n < 2) return n;
  long x, y;
  x = fib(n-1);
  y = fib(n-2);
  return x+y;
}
