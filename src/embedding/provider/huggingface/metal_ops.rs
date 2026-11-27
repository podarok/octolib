use candle_core::backend::BackendStorage;
use candle_core::{bail, CpuStorage, CustomOp1, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void merge_heads_half4(device const half4* src [[buffer(0)]], device half4* dst [[buffer(1)]],
                              constant uint4& dims [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint b = dims.x, h = dims.y, l = dims.z, d4 = dims.w;
    if (gid >= b * h * l * d4) return;
    uint di = gid % d4; uint t = gid / d4;
    uint hi = t % h; t /= h;
    uint li = t % l; uint bi = t / l;
    dst[gid] = src[((bi * h + hi) * l + li) * d4 + di];
}
kernel void bias_add_half4(device const half4* y [[buffer(0)]], device const half4* bias [[buffer(1)]],
                           device const half4* res [[buffer(2)]], device half4* dst [[buffer(3)]],
                           constant uint3& dims [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint n = dims.x, cols4 = dims.y, mode = dims.z;
    if (gid >= n) return;
    float4 v = float4(y[gid]) + float4(bias[gid % cols4]);
    if (mode == 1) v += float4(res[gid]);
    if (mode == 2) {
        // gelu_erf with the A&S 7.1.26 erf candle uses
        float4 x = abs(v * M_SQRT1_2_F);
        float4 t = 1.0 / (1.0 + 0.3275911 * x);
        float4 e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
        v = v * (1.0 + sign(v) * e) * 0.5;
    }
    dst[gid] = half4(v);
}
"#;

type Cell = std::sync::OnceLock<std::result::Result<ComputePipeline, String>>;
static MERGE: Cell = Cell::new();
static BIAS: Cell = Cell::new();

fn pipeline<'a>(cell: &'a Cell, device: &candle_core::MetalDevice, name: &str) -> Result<&'a ComputePipeline> {
	cell.get_or_init(|| {
		let dev = device.metal_device();
		let lib = dev.new_library_with_source(SRC, None).map_err(|e| e.to_string())?;
		let f = lib.get_function(name, None).map_err(|e| e.to_string())?;
		dev.new_compute_pipeline_state_with_function(&f).map_err(|e| e.to_string())
	})
	.as_ref()
	.map_err(|e| candle_core::Error::Msg(e.clone()))
}

fn grid(threads: usize) -> (MTLSize, MTLSize) {
	(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: 256.min(threads.max(1)), height: 1, depth: 1 })
}

struct MergeHeads;

impl CustomOp1 for MergeHeads {
	fn name(&self) -> &'static str {
		"merge-heads"
	}

	fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
		bail!("merge-heads: Metal only")
	}

	fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
		let &[b, h, l, d] = layout.dims() else { bail!("merge-heads: rank 4 expected") };
		if storage.dtype() != DType::F16 || !layout.is_contiguous() || d % 4 != 0 {
			bail!("merge-heads: contiguous f16 with d % 4 == 0 expected");
		}
		let device = storage.device();
		let pipeline = pipeline(&MERGE, device, "merge_heads_half4")?;
		let n = b * h * l * d;
		let out = device.new_buffer_builder().with_size_for(n, DType::F16).with_label("merge-heads").build()?;
		let guard = device.command_encoder()?;
		let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
		enc.set_compute_pipeline_state(pipeline);
		enc.set_input_buffer(0, Some(storage.buffer()), layout.start_offset() * 2);
		enc.set_output_buffer(1, Some(&out), 0);
		let dims = [b as u32, h as u32, l as u32, (d / 4) as u32];
		enc.set_bytes(2, &dims);
		let (g, t) = grid(n / 4);
		enc.dispatch_threads(g, t);
		drop(guard);
		Ok((MetalStorage::new(out, device.clone(), n, DType::F16), Shape::from((b, l, h, d))))
	}
}

struct BiasAdd {
	mode: u32,
}

impl CustomOp3 for BiasAdd {
	fn name(&self) -> &'static str {
		"bias-add"
	}

	fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
		bail!("bias-add: Metal only")
	}

	fn metal_fwd(
		&self,
		y: &MetalStorage,
		ly: &Layout,
		bias: &MetalStorage,
		lb: &Layout,
		res: &MetalStorage,
		lr: &Layout,
	) -> Result<(MetalStorage, Shape)> {
		let n = ly.shape().elem_count();
		let cols = lb.shape().elem_count();
		let ok = |s: &MetalStorage, l: &Layout| s.dtype() == DType::F16 && l.is_contiguous();
		if !ok(y, ly) || !ok(bias, lb) || !ok(res, lr) || cols % 4 != 0 || ly.dims().last() != Some(&cols) || lr.shape() != ly.shape() {
			bail!("bias-add: contiguous f16, cols % 4 == 0, bias over the last dim expected");
		}
		let device = y.device();
		let pipeline = pipeline(&BIAS, device, "bias_add_half4")?;
		let out = device.new_buffer_builder().with_size_for(n, DType::F16).with_label("bias-add").build()?;
		let guard = device.command_encoder()?;
		let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
		enc.set_compute_pipeline_state(pipeline);
		enc.set_input_buffer(0, Some(y.buffer()), ly.start_offset() * 2);
		enc.set_input_buffer(1, Some(bias.buffer()), lb.start_offset() * 2);
		enc.set_input_buffer(2, Some(res.buffer()), lr.start_offset() * 2);
		enc.set_output_buffer(3, Some(&out), 0);
		let dims = [(n / 4) as u32, (cols / 4) as u32, self.mode];
		enc.set_bytes(4, &dims);
		let (g, t) = grid(n / 4);
		enc.dispatch_threads(g, t);
		drop(guard);
		Ok((MetalStorage::new(out, device.clone(), n, DType::F16), ly.shape().clone()))
	}
}

pub fn bias_add(y: &Tensor, bias: &Tensor, res: Option<&Tensor>) -> Option<Result<Tensor>> {
	fused(y, bias, res, if res.is_some() { 1 } else { 0 })
}

pub fn bias_gelu(y: &Tensor, bias: &Tensor) -> Option<Result<Tensor>> {
	fused(y, bias, None, 2)
}

pub fn fits(t: &Tensor) -> bool {
	t.dtype() == DType::F16 && t.device().is_metal() && t.is_contiguous() && t.elem_count() > 0 && t.layout().start_offset() % 4 == 0
}

fn fused(y: &Tensor, bias: &Tensor, res: Option<&Tensor>, mode: u32) -> Option<Result<Tensor>> {
	let cols = bias.elem_count();
	if !fits(y) || !fits(bias) || bias.rank() != 1 || cols % 4 != 0 || y.dims().last() != Some(&cols) || res.is_some_and(|r| !fits(r) || r.shape() != y.shape()) {
		return None;
	}
	Some(y.apply_op3_no_bwd(bias, res.unwrap_or(y), &BiasAdd { mode }))
}

pub fn merge_heads(ctx: &Tensor) -> Result<Tensor> {
	let (b, h, l, d) = ctx.dims4()?;
	ctx.apply_op1_no_bwd(&MergeHeads)?.reshape((b, l, h * d))
}

#[cfg(test)]
mod tests {
	use candle_core::{DType, Device, Tensor};

	#[test]
	fn merge_heads_matches_transpose() -> candle_core::Result<()> {
		let Ok(dev) = Device::new_metal(0) else { return Ok(()) };
		let x = Tensor::randn(0f32, 1.0, (3, 16, 37, 64), &dev)?.to_dtype(DType::F16)?;
		let want = x.transpose(1, 2)?.reshape((3, 37, 1024))?;
		let got = super::merge_heads(&x)?;
		let diff = (want.to_dtype(DType::F32)? - got.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
		assert_eq!(diff, 0.0);
		let time = |f: &dyn Fn() -> candle_core::Result<Tensor>| -> candle_core::Result<f64> {
			f()?;
			dev.synchronize()?;
			let t = std::time::Instant::now();
			for _ in 0..50 {
				f()?;
			}
			dev.synchronize()?;
			Ok(t.elapsed().as_secs_f64() * 1e3 / 50.0)
		};
		let y = Tensor::randn(0f32, 1.0, (4, 16, 256, 64), &dev)?.to_dtype(DType::F16)?;
		println!(
			"merge b4x256: kernel {:.3} ms, strided copy {:.3} ms",
			time(&|| super::merge_heads(&y))?,
			time(&|| y.transpose(1, 2)?.contiguous())?
		);
		let z = Tensor::randn(0f32, 1.0, (1024, 1024), &dev)?.to_dtype(DType::F16)?;
		let r = Tensor::randn(0f32, 1.0, (1024, 1024), &dev)?.to_dtype(DType::F16)?;
		let bias = Tensor::randn(0f32, 1.0, 1024, &dev)?.to_dtype(DType::F16)?;
		let plain = |res: bool| -> candle_core::Result<Tensor> {
			let s = (&z + bias.broadcast_as(z.shape())?.contiguous()?)?;
			if res { s + &r } else { Ok(s) }
		};
		for res in [false, true] {
			let got = super::bias_add(&z, &bias, res.then_some(&r)).expect("fits")?;
			let d = (plain(res)?.to_dtype(DType::F32)? - got.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
			assert!(d <= 0.01, "bias-add res={res} max|diff|={d}");
			println!(
				"bias-add res={res} 1024x1024: kernel {:.3} ms, expand+add {:.3} ms, max|diff| {d}",
				time(&|| super::bias_add(&z, &bias, res.then_some(&r)).expect("fits"))?,
				time(&|| plain(res))?
			);
		}
		let wide = Tensor::randn(0f32, 1.0, 1028, &dev)?.to_dtype(DType::F16)?;
		assert!(super::bias_add(&z, &wide.narrow(0, 2, 1024)?, None).is_none(), "unaligned bias must fall back");
		assert!(super::bias_add(&z, &bias.reshape((4, 256))?, None).is_none(), "2-D bias must fall back");
		let empty = Tensor::zeros((0, 1024), DType::F16, &dev)?;
		assert!(super::bias_add(&empty, &bias, None).is_none(), "empty input must fall back");
		let got = super::bias_gelu(&z, &bias).expect("fits")?;
		let want = plain(false)?.gelu_erf()?;
		let d = (want.to_dtype(DType::F32)? - got.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
		assert!(d <= 0.01, "bias-gelu max|diff|={d}");
		println!(
			"bias-gelu 1024x1024: kernel {:.3} ms, expand+add+gelu {:.3} ms, max|diff| {d}",
			time(&|| super::bias_gelu(&z, &bias).expect("fits"))?,
			time(&|| plain(false)?.gelu_erf())?
		);
		Ok(())
	}
}
