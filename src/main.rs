#![allow(dead_code, unused_imports, unused_variables)]
#![feature(portable_simd)]
use array_macro::array;
use clap::{App, Arg, ArgMatches};
use core_simd::*;
use std::any::Any;
use std::convert::TryInto;
use std::env;
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests;

struct BmarkConfig {
  mat_dim: usize,
  accel: String,
  iters: usize,
}

impl BmarkConfig {
  fn new(mat_dim: usize, accel: String, iters: usize) -> Self {
    Self {
      mat_dim: mat_dim,
      accel: accel,
      iters: iters,
    }
  }
}

fn parse_args() -> BmarkConfig {
  let matches = App::new("MatMul")
    .arg(
      Arg::with_name("mat_size")
        .short("s")
        .long("size")
        .value_name("MATSIZE")
        .help("Dim on NxN matrix, default 128, ")
        .takes_value(true),
    )
    .arg(
      Arg::with_name("accel")
        .short("a")
        .long("accel")
        .value_name("ACCEL")
        .help("Accelerator type, default none")
        .takes_value(true),
    )
    .arg(
      Arg::with_name("itters")
        .short("i")
        .long("itters")
        .value_name("NUM_ITTERS")
        .help("Number of itterations, default 10")
        .takes_value(true),
    )
    .get_matches();

  let mat_size_str = matches.value_of("mat_size").unwrap_or("128");
  let accel_str = matches.value_of("accel").unwrap_or("none");
  let iters_str = matches.value_of("itters").unwrap_or("10");

  println!("args: -s {}, -a {}", mat_size_str, accel_str);

  let mat_size: usize = mat_size_str
    .parse::<usize>()
    .expect("-s does not parse to u32");
  let iters: usize = iters_str
    .parse::<usize>()
    .expect("-i does not parse to u32");

  BmarkConfig::new(mat_size, accel_str.to_string(), iters)
}

enum KernelType {
  F32Vec(Vec<f32>),
  F32x4Vec(Vec<f32x4>),
}

impl KernelType {
  fn len(&self) -> usize {
    match self {
      KernelType::F32Vec(v) => v.len(),
      KernelType::F32x4Vec(v) => v.len(),
    }
  }
}

fn sum_serial(a: &KernelType, b: &KernelType, dim: usize) -> KernelType {
  let mut c: Vec<f32> = vec![0.0; (dim * dim) as usize];
  if let (KernelType::F32Vec(inner_a), KernelType::F32Vec(inner_b)) = (a, b) {
    for i in 0..(dim * dim) as usize {
      c[i] = inner_a[i] + inner_b[i]
    }
  } else {
    panic!("KernelType::F32Vec type coercion failed in sum_f32x4() kernel")
  }
  KernelType::F32Vec(c)
}

fn sum_f32x4(a: &KernelType, b: &KernelType, dim: usize) -> KernelType {
  let mut c: Vec<f32x4> = Vec::with_capacity(a.len());
  if let (KernelType::F32x4Vec(inner_a), KernelType::F32x4Vec(inner_b)) = (a, b) {
    for i in 0..a.len() {
      c.push(inner_a[i] + inner_b[i]);
    }
  } else {
    panic!("KernelType::F32x4Vec type coercion failed in sum_f32x4() kernel")
  }
  KernelType::F32x4Vec(c)
}

fn vec_to_f32x4(v: &Vec<f32>) -> Vec<f32x4> {
  if v.len() % 4 != 0 {
    panic!("Dim of square matric must be divisible by 2");
  }

  let mut v_out: Vec<f32x4> = Vec::with_capacity(v.len() / 4);
  for i in 0..(v.len() / 4) {
    let tmp: [f32; 4] = v[i * 4..i * 4 + 4].try_into().unwrap();
    v_out.push(f32x4::from_array(tmp));
  }
  v_out
}

fn f32x4_to_vec(v: &Vec<f32x4>) -> Vec<f32> {
  let mut v_out: Vec<f32> = Vec::with_capacity(v.len() * 4);
  for e in v.iter() {
    v_out.append(&mut e.to_array().to_vec());
  }
  v_out
}

fn build_vec(dim: usize) -> Vec<f32> {
  let v: Vec<f32> = (0..(dim * dim)).map(|x| x as f32).collect();
  v
}

fn main() {
  let bm_cfg = parse_args();

  let a: Vec<f32> = build_vec(bm_cfg.mat_dim);
  let b: Vec<f32> = build_vec(bm_cfg.mat_dim);

  let simd_a = vec_to_f32x4(&a);
  let simd_b = vec_to_f32x4(&b);

  let kt_a = KernelType::F32Vec(a);
  let kt_b = KernelType::F32Vec(b);
  let kt_simd_a = KernelType::F32x4Vec(simd_a);
  let kt_simd_b = KernelType::F32x4Vec(simd_b);

  let tst_fn = match &bm_cfg.accel[..] {
    "none" => sum_serial,
    "f32x4" => sum_f32x4,
    _ => panic!("-a flag not matches, options are 'none', 'f32x4'"),
  };

  let (p1, p2) = match &bm_cfg.accel[..] {
    "none" => (kt_a, kt_b),
    "f32x4" => (kt_simd_a, kt_simd_b),
    _ => panic!("-a flag not matches, options are 'none', 'f32x4'"),
  };

  let mut times = Vec::new();
  for i in 0..bm_cfg.iters {
    let start = Instant::now();

    let c = tst_fn(&p1, &p2, bm_cfg.mat_dim);

    let duration = start.elapsed();
    println!("Iter:{} time {} us", i, duration.as_micros());
    times.push(duration);
  }

  let total = times.iter().map(|x| x.as_micros()).sum::<u128>();

  println!(
    "Total:{} us, Avg: {} us",
    total,
    total / bm_cfg.iters as u128
  );
}
