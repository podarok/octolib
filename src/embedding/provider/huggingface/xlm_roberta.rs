
use candle_core::{DType, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Activation, Embedding, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::xlm_roberta::Config;

struct Embeddings {
	word: Embedding,
	position: Embedding,
	token_type: Embedding,
	layer_norm: LayerNorm,
	padding_idx: u32,
}

impl Embeddings {
	fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
		Ok(Self {
			word: embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?,
			position: embedding(cfg.max_position_embeddings, cfg.hidden_size, vb.pp("position_embeddings"))?,
			token_type: embedding(cfg.type_vocab_size, cfg.hidden_size, vb.pp("token_type_embeddings"))?,
			layer_norm: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?,
			padding_idx: cfg.pad_token_id,
		})
	}

	fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
		let x = (self.word.forward(input_ids)? + self.token_type.forward(token_type_ids)?)?;
		let real = input_ids.ne(self.padding_idx)?.to_dtype(DType::F32)?;
		let pos = ((real.cumsum(1)? * &real)? + self.padding_idx as f64)?.to_dtype(DType::U32)?;
		self.layer_norm.forward(&x.broadcast_add(&self.position.forward(&pos)?)?)
	}
}

struct ResNorm {
	linear: Linear,
	norm: LayerNorm,
	params: Option<Tensor>,
	eps: f32,
}

impl ResNorm {
	fn new(linear: Linear, norm: LayerNorm, eps: f64) -> Result<Self> {
		let params = match (linear.bias(), norm.bias()) {
			(Some(b), Some(nb)) => Some(Tensor::cat(&[b, norm.weight(), nb], 0)?),
			_ => None,
		};
		Ok(Self { linear, norm, params, eps: eps as f32 })
	}

	fn forward(&self, x: &Tensor, res: &Tensor) -> Result<Tensor> {
		let y = matmul(&self.linear, x)?;
		#[cfg(target_os = "macos")]
		if let Some(p) = &self.params {
			if let Some(out) = super::metal_ops::bias_res_layer_norm(&y, res, p, self.eps) {
				return out;
			}
		}
		self.norm.forward(&epilogue(&self.linear, y, Some(res))?)
	}
}

fn mix(a: &Tensor, b: &Tensor, t: f64) -> Result<Tensor> {
	(a.to_dtype(DType::F32)? * (1.0 - t))? + (b.to_dtype(DType::F32)? * t)?
}

fn mix_linear(a: &Linear, b: &Linear, t: f64) -> Result<Linear> {
	let dt = a.weight().dtype();
	let bias = match (a.bias(), b.bias()) {
		(Some(x), Some(y)) => Some(mix(x, y, t)?.to_dtype(dt)?),
		_ => None,
	};
	Ok(Linear::new(mix(a.weight(), b.weight(), t)?.to_dtype(dt)?, bias))
}

fn mix_norm(a: &LayerNorm, b: &LayerNorm, eps: f64, t: f64) -> Result<LayerNorm> {
	let dt = a.weight().dtype();
	let bias = match (a.bias(), b.bias()) {
		(Some(x), Some(y)) => mix(x, y, t)?.to_dtype(dt)?,
		_ => bail_no_bias()?,
	};
	Ok(LayerNorm::new(mix(a.weight(), b.weight(), t)?.to_dtype(dt)?, bias, eps))
}

fn bail_no_bias() -> Result<Tensor> {
	Err(candle_core::Error::Msg("layer merge needs LayerNorm biases".into()))
}

struct Layer {
	query: Linear,
	key: Linear,
	value: Linear,
	attn: ResNorm,
	inter: Linear,
	act: Activation,
	out: ResNorm,
	heads: usize,
	head_dim: usize,
}

impl Layer {
	fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
		let h = cfg.hidden_size;
		let (sa, ao) = (vb.pp("attention").pp("self"), vb.pp("attention").pp("output"));
		Ok(Self {
			query: linear(h, h, sa.pp("query"))?,
			key: linear(h, h, sa.pp("key"))?,
			value: linear(h, h, sa.pp("value"))?,
			attn: ResNorm::new(
				linear(h, h, ao.pp("dense"))?,
				layer_norm(h, cfg.layer_norm_eps, ao.pp("LayerNorm"))?,
				cfg.layer_norm_eps,
			)?,
			inter: linear(h, cfg.intermediate_size, vb.pp("intermediate").pp("dense"))?,
			act: cfg.hidden_act,
			out: ResNorm::new(
				linear(cfg.intermediate_size, h, vb.pp("output").pp("dense"))?,
				layer_norm(h, cfg.layer_norm_eps, vb.pp("output").pp("LayerNorm"))?,
				cfg.layer_norm_eps,
			)?,
			heads: cfg.num_attention_heads,
			head_dim: h / cfg.num_attention_heads,
		})
	}

	fn merged_with(&self, other: &Layer, eps: f64, t: f64) -> Result<Layer> {
		Ok(Layer {
			query: mix_linear(&self.query, &other.query, t)?,
			key: mix_linear(&self.key, &other.key, t)?,
			value: mix_linear(&self.value, &other.value, t)?,
			attn: ResNorm::new(
				mix_linear(&self.attn.linear, &other.attn.linear, t)?,
				mix_norm(&self.attn.norm, &other.attn.norm, eps, t)?,
				eps,
			)?,
			inter: mix_linear(&self.inter, &other.inter, t)?,
			act: self.act,
			out: ResNorm::new(
				mix_linear(&self.out.linear, &other.out.linear, t)?,
				mix_norm(&self.out.norm, &other.out.norm, eps, t)?,
				eps,
			)?,
			heads: self.heads,
			head_dim: self.head_dim,
		})
	}

	fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
		let (b, l, _) = x.dims3()?;
		let v = x.reshape((b, l, self.heads, self.head_dim))?.transpose(1, 2)?;
		if x.device().is_metal() {
			Ok(v)
		} else {
			v.contiguous()
		}
	}

	fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
		let (b, l, _) = x.dims3()?;
		let q = self.split_heads(&lin(&self.query, x)?)?;
		let k = self.split_heads(&lin(&self.key, x)?)?;
		let v = self.split_heads(&lin(&self.value, x)?)?;
		let scale = 1.0 / (self.head_dim as f64).sqrt();
		let ctx = if x.device().is_metal() {
			let mask = bias.broadcast_as((b, self.heads, l, l))?;
			candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.0)?
		} else {
			let scores = (q.matmul(&k.t()?)? * scale)?.broadcast_add(bias)?;
			candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?
		};
		let ctx = merge_heads(&ctx)?;
		let a = self.attn.forward(&ctx, x)?;
		let i = self.inter_act(&a)?;
		self.out.forward(&i, &a)
	}

	fn inter_act(&self, a: &Tensor) -> Result<Tensor> {
		#[cfg(target_os = "macos")]
		if let (true, Some(bias)) = (matches!(self.act, Activation::Gelu), self.inter.bias()) {
			let (b, s, k) = a.dims3()?;
			let y = a.reshape((b * s, k))?.matmul(&self.inter.weight().t()?)?.reshape((b, s, ()))?;
			if let Some(out) = super::metal_ops::bias_gelu(&y, bias) {
				return out;
			}
			return self.act.forward(&(&y + bias.broadcast_as(y.shape())?.contiguous()?)?);
		}
		self.act.forward(&lin(&self.inter, a)?)
	}
}

fn merge_heads(ctx: &Tensor) -> Result<Tensor> {
	let (b, heads, l, d) = ctx.dims4()?;
	if !ctx.device().is_metal() || !ctx.is_contiguous() || ctx.elem_count() == 0 {
		return ctx.transpose(1, 2)?.reshape((b, l, heads * d));
	}
	#[cfg(target_os = "macos")]
	if super::metal_ops::fits(ctx) && d % 4 == 0 {
		return super::metal_ops::merge_heads(ctx);
	}
	let per = (0..b)
		.map(|i| {
			let c = ctx.get(i)?;
			Tensor::cat(&(0..heads).map(|hh| c.get(hh)).collect::<Result<Vec<_>>>()?, 1)
		})
		.collect::<Result<Vec<_>>>()?;
	Tensor::stack(&per, 0)
}

fn lin(l: &Linear, x: &Tensor) -> Result<Tensor> {
	lin_res(l, x, None)
}

fn lin_res(l: &Linear, x: &Tensor, res: Option<&Tensor>) -> Result<Tensor> {
	epilogue(l, matmul(l, x)?, res)
}

fn matmul(l: &Linear, x: &Tensor) -> Result<Tensor> {
	let (b, s, k) = x.dims3()?;
	x.reshape((b * s, k))?.matmul(&l.weight().t()?)?.reshape((b, s, ()))
}

fn epilogue(l: &Linear, y: Tensor, res: Option<&Tensor>) -> Result<Tensor> {
	let Some(bias) = l.bias() else {
		return match res {
			Some(r) => y + r,
			None => Ok(y),
		};
	};
	#[cfg(target_os = "macos")]
	if let Some(out) = super::metal_ops::bias_add(&y, bias, res) {
		return out;
	}
	let y = (&y + bias.broadcast_as(y.shape())?.contiguous()?)?;
	match res {
		Some(r) => y + r,
		None => Ok(y),
	}
}

pub struct XLMRobertaEncoder {
	embeddings: Embeddings,
	layers: Vec<Layer>,
	skip: Vec<bool>,
}

impl XLMRobertaEncoder {
	pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
		let mut layers = (0..cfg.num_hidden_layers)
			.map(|i| Layer::load(cfg, vb.pp(format!("encoder.layer.{i}"))))
			.collect::<Result<Vec<_>>>()?;
		let mut skip = vec![false; layers.len()];
		if let Ok(spec) = std::env::var("OCTOLIB_XLMR_MERGE_LAYERS") {
			for pair in spec.split(',') {
				let mut it = pair.split(':').filter_map(|s| s.trim().parse::<usize>().ok());
				if let (Some(a), Some(b)) = (it.next(), it.next()) {
					if a != b && (1..=layers.len()).contains(&a) && (1..=layers.len()).contains(&b) {
						layers[a - 1] = layers[a - 1].merged_with(&layers[b - 1], cfg.layer_norm_eps, 0.5)?;
						skip[b - 1] = true;
					}
				}
			}
		}
		if let Ok(list) = std::env::var("OCTOLIB_XLMR_SKIP_LAYERS") {
			for n in list.split(',').filter_map(|s| s.trim().parse::<usize>().ok()) {
				if (1..=skip.len()).contains(&n) {
					skip[n - 1] = true;
				}
			}
		}
		Ok(Self { embeddings: Embeddings::load(cfg, vb.pp("embeddings"))?, layers, skip })
	}

	pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
		let token_type_ids = input_ids.zeros_like()?;
		let mut x = self.embeddings.forward(input_ids, &token_type_ids)?;
		let (b, l) = mask.dims2()?;
		let bias = ((mask.to_dtype(DType::F32)? - 1.0)? * f32::MAX as f64)?
			.reshape((b, 1, 1, l))?
			.to_dtype(x.dtype())?;
		for (layer, skip) in self.layers.iter().zip(&self.skip) {
			if !skip {
				x = layer.forward(&x, &bias)?;
			}
		}
		Ok(x)
	}
}
