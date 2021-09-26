# rs_simd_bm
Was interested in rust core_simd, wrote some benchmarks

Gonna need nightly channel to run.

There are 3 command line flags, -s size, -a accelerator, -i itteratoins

running `cargo r -- -s 1024 -a f32x4 -i 20` would perform:
  - 1024x1024 matrix
  - use f32x4 
  - 20 itteratoins
