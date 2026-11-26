// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Encoder-only XLM-RoBERTa for embeddings.
//!
//! Same weights and math as `candle_transformers::models::xlm_roberta::XLMRobertaModel`, without
//! cross-attention, KV cache or heads. The padding mask is built once per forward as a
//! `(batch, 1, 1, seq)` additive bias in the model dtype and broadcast in every layer; candle's
//! model builds an f32 `(batch, 1, seq, seq)` mask and casts it to the scores dtype in each of the
//! layers.

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
		// Positions count real tokens from padding_idx + 1; pads keep padding_idx (HF
		// create_position_ids_from_input_ids). Counted in u32/f32, exact at any length.
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
		x.reshape((b, l, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()
	}

	fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
		let (b, l, h) = x.dims3()?;
		let q = self.split_heads(&self.query.forward(x)?)?;
		let k = self.split_heads(&self.key.forward(x)?)?;
		let v = self.split_heads(&self.value.forward(x)?)?;
		let scale = 1.0 / (self.head_dim as f64).sqrt();
		let ctx = if x.device().is_metal() {
			// Fused Metal kernel; the mask is a stride-0 view, never materialised per head.
			let mask = bias.broadcast_as((b, self.heads, l, l))?;
			candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.0)?
		} else {
			let scores = (q.matmul(&k.t()?)? * scale)?.broadcast_add(bias)?;
			candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?
		};
		let ctx = ctx.transpose(1, 2)?.reshape((b, l, h))?;
		let a = self.attn_norm.forward(&(self.attn_out.forward(&ctx)? + x)?)?;
		let i = self.act.forward(&self.inter.forward(&a)?)?;
		self.out_norm.forward(&(self.out.forward(&i)? + &a)?)
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

	/// `mask` is `(batch, seq)` with 1 for real tokens and 0 for padding, any dtype.
	/// Returns the last hidden state `(batch, seq, hidden)`.
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
