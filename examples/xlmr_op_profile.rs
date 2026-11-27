use candle_core::{DType, Device, Tensor, D};

fn time(dev: &Device, iters: usize, f: &dyn Fn() -> candle_core::Result<Tensor>) -> candle_core::Result<f64> {
    for _ in 0..3 {
        f()?;
    }
    dev.synchronize()?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    dev.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn main() -> candle_core::Result<()> {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (b, l) = (args.first().copied().unwrap_or(4), args.get(1).copied().unwrap_or(256));
    let (h, heads, ff) = (1024usize, 16usize, 4096usize);
    let dev = Device::new_metal(0)?;
    let r = |s: &[usize]| Tensor::randn(0f32, 0.02, s, &dev)?.to_dtype(DType::F16);
    let x = r(&[b * l, h])?;
    let w_h = r(&[h, h])?;
    let w_qkv = r(&[3 * h, h])?;
    let w_up = r(&[ff, h])?;
    let w_down = r(&[h, ff])?;
    let bias_h = r(&[h])?;
    let up = r(&[b * l, ff])?;
    let q = r(&[b, heads, l, h / heads])?;
    let mask = Tensor::zeros((b, 1, 1, l), DType::F16, &dev)?.broadcast_as((b, heads, l, l))?;
    let gamma = r(&[h])?;
    let bias_full = bias_h.broadcast_as((b * l, h))?.contiguous()?;
    let bias_ff = r(&[ff])?;
    let w_up_t = w_up.t()?.contiguous()?;
    let ones = Tensor::ones((b * l, 1), DType::F16, &dev)?;
    let bias_row = bias_h.reshape((1, h))?;
    {
        let v = x.reshape((b, l, heads, h / heads))?.transpose(1, 2)?;
        let a = candle_nn::ops::sdpa(&v, &v, &v, Some(&mask), false, 0.125, 1.0)?;
        let c = v.contiguous()?;
        let e = candle_nn::ops::sdpa(&c, &c, &c, Some(&mask), false, 0.125, 1.0)?;
        let diff = (a.to_dtype(DType::F32)? - e.to_dtype(DType::F32)?)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let bias_ok = ((&x + ones.matmul(&bias_row)?)? - x.broadcast_add(&bias_h)?)?
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        println!("sdpa view vs contiguous max|diff|={diff:e}; rank-1 bias vs broadcast max|diff|={bias_ok:e}");
    }
    let iters = 50;
    let gemm = |n: usize, k: usize| 2.0 * (b * l) as f64 * n as f64 * k as f64;
    let rows: Vec<(&str, f64, f64)> = vec![
        ("q/k/v/o proj (x4 per layer)", time(&dev, iters, &|| x.matmul(&w_h.t()?))?, gemm(h, h)),
        ("fused qkv proj", time(&dev, iters, &|| x.matmul(&w_qkv.t()?))?, gemm(3 * h, h)),
        ("ffn up", time(&dev, iters, &|| x.matmul(&w_up.t()?))?, gemm(ff, h)),
        ("ffn down", time(&dev, iters, &|| up.matmul(&w_down.t()?))?, gemm(h, ff)),
        ("bias add (h)", time(&dev, iters, &|| x.broadcast_add(&bias_h))?, 0.0),
        ("gelu (ff)", time(&dev, iters, &|| up.gelu_erf())?, 0.0),
        ("layer_norm", time(&dev, iters, &|| candle_nn::ops::layer_norm(&x, &gamma, &bias_h, 1e-5))?, 0.0),
        (
            "sdpa",
            time(&dev, iters, &|| candle_nn::ops::sdpa(&q, &q, &q, Some(&mask), false, 0.125, 1.0))?,
            4.0 * (b * heads * l * l * (h / heads)) as f64,
        ),
        ("split heads copy", time(&dev, iters, &|| x.reshape((b, l, heads, h / heads))?.transpose(1, 2)?.contiguous())?, 0.0),
        ("softmax-free sum check", time(&dev, iters, &|| x.sum_keepdim(D::Minus1))?, 0.0),
        ("add same-shape contiguous", time(&dev, iters, &|| &x + &x)?, 0.0),
        ("bias pre-broadcast contig", time(&dev, iters, &|| &x + &bias_full)?, 0.0),
        ("bias via broadcast_as", time(&dev, iters, &|| &x + &bias_h.broadcast_as(x.shape())?)?, 0.0),
        ("x.affine (scale only)", time(&dev, iters, &|| x.affine(1.0, 0.0))?, 0.0),
        ("sdpa on transposed view", time(&dev, iters, &|| {
            let v = x.reshape((b, l, heads, h / heads))?.transpose(1, 2)?;
            candle_nn::ops::sdpa(&v, &v, &v, Some(&mask), false, 0.125, 1.0)
        })?, 0.0),
        ("ffn up + bias", time(&dev, iters, &|| x.matmul(&w_up.t()?)?.broadcast_add(&bias_ff))?, 0.0),
        ("bias expand copy only", time(&dev, iters, &|| bias_h.broadcast_as(x.shape())?.contiguous())?, 0.0),
        ("bias rank-1 gemm", time(&dev, iters, &|| ones.matmul(&bias_row))?, 0.0),
        ("bias rank-1 gemm + add", time(&dev, iters, &|| &x + ones.matmul(&bias_row)?)?, 0.0),
        ("w.t() contiguous once", time(&dev, iters, &|| x.matmul(&w_up_t))?, gemm(ff, h)),
    ];
    println!("batch={b} len={l} tokens={}", b * l);
    let mut layer = 0.0;
    for (name, ms, flops) in &rows {
        let tf = if *flops > 0.0 { format!("{:.2} TFLOPS", flops / ms / 1e9) } else { String::new() };
        println!("{name:<28} {ms:>8.3} ms  {tf}");
    }
    for (name, ms, _) in &rows {
        layer += match *name {
            "q/k/v/o proj (x4 per layer)" => 4.0 * ms,
            "ffn up" | "ffn down" | "gelu (ff)" | "sdpa" => *ms,
            "bias add (h)" => 4.0 * ms,
            "layer_norm" => 2.0 * ms,
            "split heads copy" => 4.0 * ms,
            _ => 0.0,
        };
    }
    println!("layer estimate {layer:.2} ms, model (24 layers) {:.1} ms = {:.1} chunks/s", layer * 24.0, b as f64 / (layer * 24.0 / 1e3));
    Ok(())
}
