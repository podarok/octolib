//! Measured roofline of weight-only low-bit GEMM on the M3 Pro (epic-20 P20.47, E20-D66).
//!
//! 1. DRAM bandwidth: an own streaming-read kernel over a 512 MB buffer (candle's sum_all is not a bandwidth probe);
//! 2. the layer GEMMs (N,K) = (1024,1024) x4, (4096,1024), (1024,4096) per layer, 24 layers, for M = 25 (query) and M = 600
//!    (bulk), each call on a DIFFERENT weight matrix from a set larger than the system cache (the model streams 600 MB of
//!    weights per query), so a call pays the DRAM read like the model does;
//! 3. candle f16 matmul vs an own simdgroup_matrix kernel (32x32 tile, weights unpacked to half in registers / threadgroup
//!    memory): f16 (kernel-quality control), int8, int4, int2, ternary (2-bit store), 1-bit; scale per group of 64;
//!    every kernel is checked against a CPU reference;
//! 4. per-model totals (24 layers) for query and bulk, next to the measured 13.6 ms query encode.
//! Log: <eval>/logs/lowbit-roofline.log (logfmt, PKB tools policy).
//! Run: cargo run --release --features huggingface --example lowbit_roofline
use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

use candle_core::backend::BackendStorage;
use candle_core::{bail, CpuStorage, CustomOp1, CustomOp3, DType, Device, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;

const M_QUERY: usize = 25;
const M_BULK: usize = 600;
const LAYERS: usize = 24;

fn log(level: &str, event: &str, fields: &str) {
    let ts = std::process::Command::new("date").arg("+%FT%T").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let line = format!("ts={ts} level={level} run=lowbit-roofline event={event} {fields}");
    eprintln!("{line}");
    if let Some(home) = std::env::var_os("HOME") {
        let p = std::path::Path::new(&home).join("Library/Application Support/ec-chats/eval/logs/lowbit-roofline.log");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn mm_src(bits: usize, kind: usize) -> String {
    format!(
        r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define BITS {bits}
#define KIND {kind}
kernel void lowbit_mm(device const half* x [[buffer(0)]], device const uint* wp [[buffer(1)]], device const half* sc [[buffer(2)]],
                      device half* out [[buffer(3)]], constant uint3& dims [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {{
    uint K = dims.x, N = dims.y, M = dims.z;
    uint n0 = tg.x * 32, m0 = tg.y * 32;
    threadgroup half xs[32 * 32];
    threadgroup half ws[32 * 32];
    threadgroup float tmp[32 * 32];
    simdgroup_float8x8 acc[4];
    for (uint i = 0; i < 4; i++) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint kw = K * BITS / 32;
    uint groups = K / 64;
    uint row = tid / 4, part = (tid % 4) * 8;
    for (uint k0 = 0; k0 < K; k0 += 32) {{
        uint m = m0 + row;
        for (uint v = 0; v < 8; v++) xs[row * 32 + part + v] = (m < M) ? x[m * K + k0 + part + v] : half(0);
        uint n = n0 + row;
#if KIND == 3
        device const half* w16 = (device const half*)wp;
        for (uint v = 0; v < 8; v++) ws[row * 32 + part + v] = w16[n * K + k0 + part + v];
#else
        float s = float(sc[n * groups + (k0 + part) / 64]);
        for (uint v = 0; v < 8; v++) {{
            uint bitpos = (k0 + part + v) * BITS;
            uint q = (wp[n * kw + (bitpos >> 5)] >> (bitpos & 31)) & ((1u << BITS) - 1u);
#if KIND == 1
            float w = (q != 0 ? 1.0f : -1.0f) * s;
#elif KIND == 2
            float w = (float(q) - 1.0f) * s;
#else
            float w = (float(q) - float(1u << (BITS - 1))) * s;
#endif
            ws[row * 32 + part + v] = half(w);
        }}
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 4; kk++) {{
            simdgroup_half8x8 b;
            simdgroup_load(b, ws + (sg * 8) * 32 + kk * 8, 32, ulong2(0, 0), true);
            for (uint tb = 0; tb < 4; tb++) {{
                simdgroup_half8x8 a;
                simdgroup_load(a, xs + (tb * 8) * 32 + kk * 8, 32);
                simdgroup_multiply_accumulate(acc[tb], a, b, acc[tb]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (uint tb = 0; tb < 4; tb++) simdgroup_store(acc[tb], tmp + (tb * 8) * 32 + sg * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint m = m0 + row;
    if (m < M) for (uint v = 0; v < 8; v++) out[m * N + n0 + part + v] = half(tmp[row * 32 + part + v]);
}}
"#
    )
}

const BW_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void read_bw(device const float4* a [[buffer(0)]], device float* partial [[buffer(1)]], constant uint& n4 [[buffer(2)]],
                    uint gid [[thread_position_in_grid]], uint nthreads [[threads_per_grid]]) {
    float4 s = float4(0);
    for (uint i = gid; i < n4; i += nthreads) s += a[i];
    partial[gid] = s.x + s.y + s.z + s.w;
}
"#;

struct Bw(OnceLock<std::result::Result<ComputePipeline, String>>);

impl CustomOp1 for Bw {
    fn name(&self) -> &'static str {
        "read-bw"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("metal only")
    }

    fn metal_fwd(&self, a: &MetalStorage, la: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = a.device();
        let pipe = self
            .0
            .get_or_init(|| {
                let dev = device.metal_device();
                let lib = dev.new_library_with_source(BW_SRC, None).map_err(|e| e.to_string())?;
                let f = lib.get_function("read_bw", None).map_err(|e| e.to_string())?;
                dev.new_compute_pipeline_state_with_function(&f).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| candle_core::Error::Msg(e.clone()))?;
        let threads = 65536usize;
        let out = device.new_buffer_builder().with_size_for(threads, DType::F32).with_label("bw").build()?;
        let guard = device.command_encoder()?;
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(pipe);
        enc.set_input_buffer(0, Some(a.buffer()), la.start_offset() * 4);
        enc.set_output_buffer(1, Some(&out), 0);
        enc.set_bytes(2, &((la.shape().elem_count() / 4) as u32));
        enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        drop(guard);
        Ok((MetalStorage::new(out, device.clone(), threads, DType::F32), Shape::from(threads)))
    }
}

struct LowBit {
    bits: usize,
    kind: usize,
    pipe: OnceLock<std::result::Result<ComputePipeline, String>>,
}

impl CustomOp3 for LowBit {
    fn name(&self) -> &'static str {
        "lowbit-mm"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("metal only")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, wp: &MetalStorage, lwp: &Layout, sc: &MetalStorage, lsc: &Layout) -> Result<(MetalStorage, Shape)> {
        let (m, k) = lx.shape().dims2()?;
        let n = lsc.shape().dims2()?.0;
        let device = x.device();
        let pipe = self
            .pipe
            .get_or_init(|| {
                let dev = device.metal_device();
                let lib = dev.new_library_with_source(&mm_src(self.bits, self.kind), None).map_err(|e| e.to_string())?;
                let f = lib.get_function("lowbit_mm", None).map_err(|e| e.to_string())?;
                dev.new_compute_pipeline_state_with_function(&f).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| candle_core::Error::Msg(e.clone()))?;
        let out = device.new_buffer_builder().with_size_for(m * n, DType::F16).with_label("lowbit").build()?;
        let guard = device.command_encoder()?;
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(pipe);
        enc.set_input_buffer(0, Some(x.buffer()), lx.start_offset() * 2);
        enc.set_input_buffer(1, Some(wp.buffer()), lwp.start_offset() * 4);
        enc.set_input_buffer(2, Some(sc.buffer()), lsc.start_offset() * 2);
        enc.set_output_buffer(3, Some(&out), 0);
        enc.set_bytes(4, &[k as u32, n as u32, m as u32]);
        enc.dispatch_thread_groups(MTLSize { width: n / 32, height: m.div_ceil(32), depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
        drop(guard);
        Ok((MetalStorage::new(out, device.clone(), m * n, DType::F16), Shape::from((m, n))))
    }
}

struct Packed {
    words: Vec<u32>,
    scales: Vec<f32>,
    codes: Vec<u32>,
}

fn dequant(codes: &[u32], scales: &[f32], k: usize, bits: usize, kind: usize) -> Vec<f32> {
    codes
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let s = scales[(i / k) * (k / 64) + (i % k) / 64];
            match kind {
                1 => (if c != 0 { 1.0 } else { -1.0 }) * s,
                2 => (c as f32 - 1.0) * s,
                _ => (c as f32 - (1u32 << (bits - 1)) as f32) * s,
            }
        })
        .collect()
}

fn rnd(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn quantise(w: &[f32], n: usize, k: usize, bits: usize, kind: usize) -> Packed {
    let kw = k * bits / 32;
    let mut words = vec![0u32; n * kw];
    let mut scales = vec![0f32; n * (k / 64)];
    let mut codes = vec![0u32; n * k];
    for r in 0..n {
        for g in 0..k / 64 {
            let grp = &w[r * k + g * 64..r * k + g * 64 + 64];
            let amax = grp.iter().fold(0f32, |a, x| a.max(x.abs())).max(1e-8);
            let mean = grp.iter().map(|x| x.abs()).sum::<f32>() / 64.0 + 1e-8;
            let s = match kind {
                1 | 2 => mean,
                _ => amax / ((1u32 << (bits - 1)) - 1).max(1) as f32,
            };
            scales[r * (k / 64) + g] = s;
            for (i, &x) in grp.iter().enumerate() {
                let code = match kind {
                    1 => (x >= 0.0) as u32,
                    2 => ((x / s).round().clamp(-1.0, 1.0) + 1.0) as u32,
                    _ => {
                        let h = (1i32 << (bits - 1)) as f32;
                        ((x / s).round().clamp(-h, h - 1.0) + h) as u32
                    }
                };
                let kk = g * 64 + i;
                let bitpos = kk * bits;
                words[r * kw + bitpos / 32] |= code << (bitpos % 32);
                codes[r * k + kk] = code;
            }
        }
    }
    Packed { words, scales, codes }
}

/// median over 5 rounds of `iters` calls; call i uses weight set i % copies (cold cache when the set exceeds the cache)
fn time(dev: &Device, iters: usize, f: &dyn Fn(usize) -> Result<Tensor>) -> Result<f64> {
    for i in 0..8 {
        f(i)?;
    }
    dev.synchronize()?;
    let mut v = Vec::new();
    for r in 0..5 {
        let t = Instant::now();
        for i in 0..iters {
            f(r * iters + i)?;
        }
        dev.synchronize()?;
        v.push(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
    }
    v.sort_by(f64::total_cmp);
    Ok(v[2])
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let t_all = Instant::now();
    log("info", "phase_start", "phase=bandwidth");
    let big = Tensor::zeros(128 * 1024 * 1024, DType::F32, &dev)?;
    let bw = Bw(OnceLock::new());
    let ms = time(&dev, 10, &|_| big.apply_op1_no_bwd(&bw))?;
    let gbs = 512.0 * 1024.0 * 1024.0 / (ms / 1e3) / 1e9;
    log("info", "phase_end", &format!("phase=bandwidth read_512mb_ms={ms:.3} gb_s={gbs:.1}"));
    drop(big);

    // (n, k, matrices per layer)
    let shapes = [(1024usize, 1024usize, 4usize), (4096, 1024, 1), (1024, 4096, 1)];
    let schemes: [(&str, usize, usize); 6] = [("own-f16", 16, 3), ("int8", 8, 0), ("int4", 4, 0), ("int2", 2, 0), ("ternary(2-bit)", 2, 2), ("1-bit", 1, 1)];
    // totals[scheme][query|bulk] in ms per model; index 0 = candle f16
    let mut totals = vec![[0.0f64; 2]; schemes.len() + 1];
    let mut state = 0x9E3779B97F4A7C15u64;
    for &(n, k, count) in &shapes {
        let copies = (LAYERS * count).min(24 * 4);
        log("info", "phase_start", &format!("phase=shape n={n} k={k} matrices_per_layer={count} weight_sets={copies}"));
        let wf: Vec<f32> = (0..n * k).map(|_| rnd(&mut state) * 0.1).collect();
        let base = Tensor::from_vec(wf.clone(), (n, k), &dev)?.to_dtype(DType::F16)?;
        let sets: Vec<Tensor> = (0..copies).map(|_| base.copy()).collect::<Result<_>>()?;
        let sets_t: Vec<Tensor> = sets.iter().map(|w| w.t()).collect::<Result<_>>()?;
        for (mi, (m, label)) in [(M_QUERY, "query"), (M_BULK, "bulk")].into_iter().enumerate() {
            let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::F16)?;
            let t = time(&dev, 60, &|i| x.matmul(&sets_t[i % copies]))?;
            let bytes = (n * k * 2) as f64;
            log("info", "row", &format!("m={m} {label} n={n} k={k} scheme=f16-candle ms={t:.4} weight_mb={:.2} weight_read_ms_at_bw={:.4} eff_gb_s={:.1} tflops={:.2}", bytes / 1e6, bytes / (gbs * 1e9) * 1e3, bytes / (t / 1e3) / 1e9, 2.0 * (m * n * k) as f64 / t / 1e9));
            totals[0][mi] += t * (count * LAYERS) as f64;
        }
        drop(sets_t);
        drop(sets);
        for (si, &(name, bits, kind)) in schemes.iter().enumerate() {
            let (words, scales_t, deq) = if kind == 3 {
                let w16: Vec<f32> = base.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                (None, None, w16)
            } else {
                let p = quantise(&wf, n, k, bits, kind);
                let sc = Tensor::from_vec(p.scales.clone(), (n, k / 64), &dev)?.to_dtype(DType::F16)?;
                let rounded: Vec<f32> = sc.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let deq = dequant(&p.codes, &rounded, k, bits, kind);
                (Some(p.words), Some(sc), deq)
            };
            let mk = |dev: &Device| -> Result<(Tensor, Tensor)> {
                match (&words, &scales_t) {
                    (Some(w), Some(s)) => Ok((Tensor::from_vec(w.clone(), (n, k * bits / 32), dev)?, s.copy()?)),
                    _ => Ok((base.copy()?.to_dtype(DType::F16)?, Tensor::zeros((n, k / 64), DType::F16, dev)?)),
                }
            };
            let mut wsets = Vec::new();
            for _ in 0..copies {
                wsets.push(mk(&dev)?);
            }
            let op = LowBit { bits, kind, pipe: OnceLock::new() };
            let run = |x: &Tensor, i: usize| -> Result<Tensor> {
                let (w, s) = &wsets[i % copies];
                if kind == 3 {
                    // f16 weights as raw u32 words: reinterpret through a U32 tensor of the same bytes
                    x.apply_op3_no_bwd(&w.reshape((n, k))?, s, &op)
                } else {
                    x.apply_op3_no_bwd(w, s, &op)
                }
            };
            for (mi, (m, label)) in [(M_QUERY, "query"), (M_BULK, "bulk")].into_iter().enumerate() {
                let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::F16)?;
                let xh: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let got: Vec<f32> = run(&x, 0)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let mut worst = 0f32;
                for j in 0..8 {
                    for mm in 0..m.min(6) {
                        let want: f32 = (0..k).map(|kk| xh[mm * k + kk] * deq[j * k + kk]).sum();
                        worst = worst.max((got[mm * n + j] - want).abs() / (want.abs() + 1.0));
                    }
                }
                if worst > 0.03 {
                    log("warn", "gate", &format!("reason=kernel_mismatch scheme={name} m={m} n={n} k={k} rel_err={worst:.4}"));
                }
                let t = time(&dev, 60, &|i| run(&x, i))?;
                let bytes = if kind == 3 { (n * k * 2) as f64 } else { (n * k * bits / 8 + n * (k / 64) * 2) as f64 };
                log("info", "row", &format!("m={m} {label} n={n} k={k} scheme={name} ms={t:.4} weight_mb={:.3} weight_read_ms_at_bw={:.4} eff_gb_s={:.1} tflops={:.2} rel_err={worst:.4}", bytes / 1e6, bytes / (gbs * 1e9) * 1e3, bytes / (t / 1e3) / 1e9, 2.0 * (m * n * k) as f64 / t / 1e9));
                totals[si + 1][mi] += t * (count * LAYERS) as f64;
            }
        }
        log("info", "phase_end", &format!("phase=shape n={n} k={k}"));
    }
    log("info", "phase_start", "phase=totals");
    let names: Vec<&str> = std::iter::once("f16-candle").chain(schemes.iter().map(|s| s.0)).collect();
    for (i, name) in names.iter().enumerate() {
        log(
            "info",
            "total",
            &format!("scheme={name} query_m{M_QUERY}_gemm_ms_24_layers={:.2} vs_candle_f16={:+.2} bulk_m{M_BULK}_gemm_ms_24_layers={:.1} vs_candle_f16={:+.1}", totals[i][0], totals[i][0] - totals[0][0], totals[i][1], totals[i][1] - totals[0][1]),
        );
    }
    log("info", "result", &format!("bandwidth_gb_s={gbs:.1} measured_query_encode_ms=13.6 seconds={:.0}", t_all.elapsed().as_secs_f64()));
    Ok(())
}
