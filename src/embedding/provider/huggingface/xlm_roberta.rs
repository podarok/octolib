
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

struct Layer {
	query: Linear,
	key: Linear,
	value: Linear,
	attn_out: Linear,
	attn_norm: LayerNorm,
	inter: Linear,
	act: Activation,
	out: Linear,
	out_norm: LayerNorm,
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
			attn_out: linear(h, h, ao.pp("dense"))?,
			attn_norm: layer_norm(h, cfg.layer_norm_eps, ao.pp("LayerNorm"))?,
			inter: linear(h, cfg.intermediate_size, vb.pp("intermediate").pp("dense"))?,
			act: cfg.hidden_act,
			out: linear(cfg.intermediate_size, h, vb.pp("output").pp("dense"))?,
			out_norm: layer_norm(h, cfg.layer_norm_eps, vb.pp("output").pp("LayerNorm"))?,
			heads: cfg.num_attention_heads,
			head_dim: h / cfg.num_attention_heads,
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
		let a = self.attn_norm.forward(&lin_res(&self.attn_out, &ctx, Some(x))?)?;
		let i = self.inter_act(&a)?;
		self.out_norm.forward(&lin_res(&self.out, &i, Some(&a))?)
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
	if !ctx.device().is_metal() || !ctx.is_contiguous() {
		return ctx.transpose(1, 2)?.reshape((b, l, heads * d));
	}
	#[cfg(target_os = "macos")]
	if ctx.dtype() == candle_core::DType::F16 && d % 4 == 0 {
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

/// x W^T + bias (+ res). On Metal f16 the bias and residual go through one kernel.
fn lin_res(l: &Linear, x: &Tensor, res: Option<&Tensor>) -> Result<Tensor> {
	let (b, s, k) = x.dims3()?;
	let y = x.reshape((b * s, k))?.matmul(&l.weight().t()?)?.reshape((b, s, ()))?;
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
}

impl XLMRobertaEncoder {
	pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
		let layers = (0..cfg.num_hidden_layers)
			.map(|i| Layer::load(cfg, vb.pp(format!("encoder.layer.{i}"))))
			.collect::<Result<Vec<_>>>()?;
		Ok(Self { embeddings: Embeddings::load(cfg, vb.pp("embeddings"))?, layers })
	}

	pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
		let token_type_ids = input_ids.zeros_like()?;
		let mut x = self.embeddings.forward(input_ids, &token_type_ids)?;
		let (b, l) = mask.dims2()?;
		let bias = ((mask.to_dtype(DType::F32)? - 1.0)? * f32::MAX as f64)?
			.reshape((b, 1, 1, l))?
			.to_dtype(x.dtype())?;
		for layer in &self.layers {
			x = layer.forward(&x, &bias)?;
		}
		Ok(x)
	}
}
