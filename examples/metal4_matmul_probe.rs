use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

use candle_core::backend::BackendStorage;
use candle_core::{bail, CpuStorage, CustomOp2, DType, Device, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLSize};

const M_QUERY: usize = 25;
const M_BULK: usize = 600;

static CONTROL_OK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn tool_rev() -> &'static str {
    static REV: OnceLock<String> = OnceLock::new();
    REV.get_or_init(|| {
        let git = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(env!("CARGO_MANIFEST_DIR")).args(args).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        let rev = git(&["rev-parse", "--short", "HEAD"]);
        let dirty = !git(&["status", "--porcelain", "--", "examples/metal4_matmul_probe.rs", "src"]).is_empty();
        format!("{rev}{}", if dirty { "-dirty" } else { "" })
    })
}

fn log(level: &str, event: &str, fields: &str) {
    let ts = std::process::Command::new("date").arg("+%FT%T").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let rev = if event == "phase_start" || event == "result" { format!(" tool_rev={}", tool_rev()) } else { String::new() };
    let line = format!("ts={ts} level={level} run=metal4-matmul-probe event={event}{rev} {fields}");
    eprintln!("{line}");
    if let Some(home) = std::env::var_os("HOME") {
        let p = std::path::Path::new(&home).join("Library/Application Support/ec-chats/eval/logs/metal4-matmul-probe.log");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn control(name: &str, value: f64, lo: f64, hi: f64) {
    let ok = value >= lo && value <= hi;
    if !ok {
        CONTROL_OK.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    log(if ok { "info" } else { "warn" }, if ok { "control" } else { "gate" }, &format!("{}control={name} value={value:.4} expected={lo}..{hi}", if ok { "" } else { "reason=control_failed " }));
}

const SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void mm2d(device half* x [[buffer(0)]], device half* w [[buffer(1)]], device half* out [[buffer(2)]],
                 constant uint3& dims [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
    int K = dims.x, N = dims.y, M = dims.z;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A(x, dextents<int32_t, 2>(K, M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(K, N));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> C(out, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(32, 32, dynamic_length_v<int>, false, true);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tg.y * 32);
    auto mB = B.slice(0, tg.x * 32);
    auto mC = C.slice(tg.x * 32, tg.y * 32);
    op.run(mA, mB, mC);
}
"#;

struct Mm2d(OnceLock<std::result::Result<ComputePipeline, String>>);

impl CustomOp2 for Mm2d {
    fn name(&self) -> &'static str {
        "mm2d"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("metal only")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, w: &MetalStorage, lw: &Layout) -> Result<(MetalStorage, Shape)> {
        let (m, k) = lx.shape().dims2()?;
        let n = lw.shape().dims2()?.0;
        let device = x.device();
        let pipe = self
            .0
            .get_or_init(|| {
                let dev = device.metal_device();
                let opts = MTLCompileOptions::new();
                opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
                let lib = dev.new_library_with_source(SRC, Some(&opts)).map_err(|e| e.to_string())?;
                let f = lib.get_function("mm2d", None).map_err(|e| e.to_string())?;
                dev.new_compute_pipeline_state_with_function(&f).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| candle_core::Error::Msg(e.clone()))?;
        let out = device.new_buffer_builder().with_size_for(m * n, DType::F16).with_label("mm2d").build()?;
        let guard = device.command_encoder()?;
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(pipe);
        enc.set_input_buffer(0, Some(x.buffer()), lx.start_offset() * 2);
        enc.set_input_buffer(1, Some(w.buffer()), lw.start_offset() * 2);
        enc.set_output_buffer(2, Some(&out), 0);
        enc.set_bytes(3, &[k as u32, n as u32, m as u32]);
        enc.dispatch_thread_groups(MTLSize { width: n / 32, height: m.div_ceil(32), depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
        drop(guard);
        Ok((MetalStorage::new(out, device.clone(), m * n, DType::F16), Shape::from((m, n))))
    }
}

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
    log("info", "phase_start", "phase=compile_and_correctness");
    let op = Mm2d(OnceLock::new());
    let (n, k) = (1024usize, 4096usize);
    let w = Tensor::randn(0f32, 0.1, (n, k), &dev)?.to_dtype(DType::F16)?;
    let mut compiled = true;
    for m in [M_QUERY, M_BULK] {
        let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::F16)?;
        match x.apply_op2_no_bwd(&w, &op) {
            Ok(got) => {
                let want = x.matmul(&w.t()?)?;
                let g: Vec<f32> = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let r: Vec<f32> = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let worst = g.iter().zip(&r).map(|(a, b)| (a - b).abs() / (b.abs() + 1.0)).fold(0f32, f32::max);
                control(&format!("mm2d_vs_candle_f16_m{m}_rel_err"), worst as f64, 0.0, 0.02);
            }
            Err(e) => {
                compiled = false;
                log("warn", "probe_failed", &format!("m={m} reason=compile_or_run error={:?}", e.to_string().chars().take(400).collect::<String>()));
                break;
            }
        }
    }
    if compiled {
        log("info", "phase_end", "phase=compile_and_correctness compiled=true");
        for (n, k) in [(1024usize, 1024usize), (1024, 4096), (4096, 1024)] {
            let w = Tensor::randn(0f32, 0.1, (n, k), &dev)?.to_dtype(DType::F16)?;
            let wt = w.t()?;
            for (m, label) in [(M_QUERY, "query"), (M_BULK, "bulk")] {
                let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::F16)?;
                let flop = 2.0 * (m * n * k) as f64;
                let tc = time(&dev, 60, &|_| x.matmul(&wt))?;
                let tm = time(&dev, 60, &|_| x.apply_op2_no_bwd(&w, &op))?;
                log("info", "row", &format!("m={m} {label} n={n} k={k} candle_ms={tc:.4} candle_tflops={:.2} mm2d_ms={tm:.4} mm2d_tflops={:.2} speedup={:.2}", flop / tc / 1e9, flop / tm / 1e9, tc / tm));
                if (n, k, m) == (1024, 4096, M_BULK) {
                    control("candle_f16_tflops_m600_1024x4096_vs_profile_5.2", flop / tc / 1e9, 4.4, 5.8);
                }
            }
        }
    }
    let ok = CONTROL_OK.load(std::sync::atomic::Ordering::Relaxed);
    log(if ok { "info" } else { "warn" }, "result", &format!("controls={} compiled={compiled} seconds={:.0}", if ok { "ok" } else { "failed_numbers_not_results" }, t_all.elapsed().as_secs_f64()));
    if !ok {
        std::process::exit(2);
    }
    Ok(())
}
