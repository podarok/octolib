use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

use candle_core::backend::BackendStorage;
use candle_core::{bail, CpuStorage, CustomOp2, DType, Device, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLSize};

static CONTROL_OK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn tool_rev() -> &'static str {
    static REV: OnceLock<String> = OnceLock::new();
    REV.get_or_init(|| {
        let git = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(env!("CARGO_MANIFEST_DIR")).args(args).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        let dirty = !git(&["status", "--porcelain", "--", "examples/metal4_int8_probe.rs", "src"]).is_empty();
        format!("{}{}", git(&["rev-parse", "--short", "HEAD"]), if dirty { "-dirty" } else { "" })
    })
}

fn log(level: &str, event: &str, fields: &str) {
    let ts = std::process::Command::new("date").arg("+%FT%T").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let rev = if event == "phase_start" || event == "result" { format!(" tool_rev={}", tool_rev()) } else { String::new() };
    let line = format!("ts={ts} level={level} run=metal4-int8-probe event={event}{rev} {fields}");
    eprintln!("{line}");
    if let Some(home) = std::env::var_os("HOME") {
        let p = std::path::Path::new(&home).join("Library/Application Support/ec-chats/eval/logs/metal4-int8-probe.log");
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

fn src(a: &str, b: &str, c: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void mm(device {a}* x [[buffer(0)]], device {b}* w [[buffer(1)]], device {c}* out [[buffer(2)]],
               constant packed_uint3& dims [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {{
    int K = dims.x, N = dims.y, M = dims.z;
    tensor<device {a}, dextents<int32_t, 2>, tensor_inline> A(x, dextents<int32_t, 2>(K, M));
    tensor<device {b}, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(K, N));
    tensor<device {c}, dextents<int32_t, 2>, tensor_inline> C(out, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(32, 32, dynamic_length_v<int>, false, true);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tg.y * 32);
    auto mB = B.slice(0, tg.x * 32);
    auto mC = C.slice(tg.x * 32, tg.y * 32);
    op.run(mA, mB, mC);
}}
"#
    )
}

struct Mm {
    source: String,
    out_dtype: DType,
    pipe: OnceLock<std::result::Result<ComputePipeline, String>>,
}

impl CustomOp2 for Mm {
    fn name(&self) -> &'static str {
        "mm"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("metal only")
    }

    fn metal_fwd(&self, x: &MetalStorage, lx: &Layout, w: &MetalStorage, lw: &Layout) -> Result<(MetalStorage, Shape)> {
        let (m, k) = lx.shape().dims2()?;
        let n = lw.shape().dims2()?.0;
        let device = x.device();
        let pipe = self
            .pipe
            .get_or_init(|| {
                let dev = device.metal_device();
                let opts = MTLCompileOptions::new();
                opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
                let lib = dev.new_library_with_source(&self.source, Some(&opts)).map_err(|e| e.to_string())?;
                let f = lib.get_function("mm", None).map_err(|e| e.to_string())?;
                dev.new_compute_pipeline_state_with_function(&f).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| candle_core::Error::Msg(e.clone()))?;
        let out = device.new_buffer_builder().with_size_for(m * n, self.out_dtype).with_label("mm").build()?;
        let esz = x.dtype().size_in_bytes();
        let wsz = w.dtype().size_in_bytes();
        let guard = device.command_encoder()?;
        let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(pipe);
        enc.set_input_buffer(0, Some(x.buffer()), lx.start_offset() * esz);
        enc.set_input_buffer(1, Some(w.buffer()), lw.start_offset() * wsz);
        enc.set_output_buffer(2, Some(&out), 0);
        enc.set_bytes(3, &[k as u32, n as u32, m as u32]);
        enc.dispatch_thread_groups(MTLSize { width: n / 32, height: m.div_ceil(32), depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
        drop(guard);
        Ok((MetalStorage::new(out, device.clone(), m * n, self.out_dtype), Shape::from((m, n))))
    }
}

fn time(dev: &Device, iters: usize, f: &dyn Fn() -> Result<Tensor>) -> Result<f64> {
    for _ in 0..8 {
        f()?;
    }
    dev.synchronize()?;
    let mut v = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            f()?;
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
    log("info", "phase_start", "phase=probe");
    let cases: [(&str, &str, &str, &str, DType); 3] = [
        ("half x half -> half (control)", "half", "half", "half", DType::F16),
        ("int8 x int8 -> int32", "char", "char", "int", DType::U32),
        ("half x int8 -> half", "half", "char", "half", DType::F16),
    ];
    let (n, k) = (1024usize, 4096usize);
    let mut ok_cases = Vec::new();
    for (name, a, b, c, odt) in cases {
        let op = Mm { source: src(a, b, c), out_dtype: odt, pipe: OnceLock::new() };
        let xdt = if a == "half" { DType::F16 } else { DType::U8 };
        let wdt = if b == "half" { DType::F16 } else { DType::U8 };
        let mk = |rows: usize, cols: usize, dt: DType| -> Result<Tensor> {
            if dt == DType::F16 {
                Tensor::randn(0f32, 0.5, (rows, cols), &dev)?.to_dtype(DType::F16)
            } else {
                let v: Vec<u8> = (0..rows * cols).map(|i| (((i * 2654435761usize) >> 13) % 7) as u8).collect();
                Tensor::from_vec(v, (rows, cols), &dev)
            }
        };
        let w = mk(n, k, wdt)?;
        let m_check = 25;
        let x = mk(m_check, k, xdt)?;
        match x.apply_op2_no_bwd(&w, &op) {
            Err(e) => {
                log("warn", "case_failed", &format!("case={} reason=compile_or_run error={:?}", name.replace(' ', "_"), e.to_string().chars().take(300).collect::<String>()));
                continue;
            }
            Ok(got) => {
                if a == "char" && b == "char" {
                    let xv: Vec<u8> = x.flatten_all()?.to_vec1()?;
                    let wv: Vec<u8> = w.flatten_all()?.to_vec1()?;
                    let gv: Vec<u32> = got.flatten_all()?.to_vec1()?;
                    let mut bad = 0usize;
                    for mm in 0..3 {
                        for j in 0..16 {
                            let want: i32 = (0..k).map(|kk| (xv[mm * k + kk] as i8 as i32) * (wv[j * k + kk] as i8 as i32)).sum();
                            if gv[mm * n + j] as i32 != want {
                                bad += 1;
                            }
                        }
                    }
                    control("int8_vs_cpu_integer_reference_mismatches", bad as f64, 0.0, 0.0);
                } else if a == "half" && b == "half" {
                    let want = x.matmul(&w.t()?)?;
                    let d = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                    control("half_vs_candle_max_abs_diff", d as f64, 0.0, 0.1);
                } else {
                    let wf = w.to_dtype(DType::F32)?;
                    let _ = wf;
                }
            }
        }
        ok_cases.push((name, op, xdt, wdt));
        log("info", "case_ok", &format!("case={}", name.replace(' ', "_")));
    }
    log("info", "phase_end", &format!("phase=probe cases_ok={}", ok_cases.len()));
    for (name, op, xdt, wdt) in &ok_cases {
        for (n, k) in [(1024usize, 1024usize), (1024, 4096), (4096, 1024)] {
            let mk = |rows: usize, cols: usize, dt: DType| -> Result<Tensor> {
                if dt == DType::F16 {
                    Tensor::randn(0f32, 0.5, (rows, cols), &dev)?.to_dtype(DType::F16)
                } else {
                    Tensor::from_vec((0..rows * cols).map(|i| ((i >> 3) % 5) as u8).collect::<Vec<u8>>(), (rows, cols), &dev)
                }
            };
            let w = mk(n, k, *wdt)?;
            for (m, label) in [(25usize, "query"), (256, "mid"), (600, "bulk")] {
                let x = mk(m, k, *xdt)?;
                let t = time(&dev, 60, &|| x.apply_op2_no_bwd(&w, op))?;
                let flop = 2.0 * (m * n * k) as f64;
                log("info", "row", &format!("case={} m={m} {label} n={n} k={k} ms={t:.4} tops={:.2}", name.replace(' ', "_"), flop / t / 1e9));
            }
        }
    }
    let ok = CONTROL_OK.load(std::sync::atomic::Ordering::Relaxed);
    log(if ok { "info" } else { "warn" }, "result", &format!("controls={} cases_ok={} seconds={:.0}", if ok { "ok" } else { "failed_numbers_not_results" }, ok_cases.len(), t_all.elapsed().as_secs_f64()));
    if !ok {
        std::process::exit(2);
    }
    Ok(())
}
