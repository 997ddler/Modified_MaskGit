import torch
import torch.nn as nn
import torch.nn.functional as F

from semivq.dists import get_dist_fns
import semivq
from semivq.norms import with_codebook_normalization
from .vq_base import _VQBaseLayer
from .affine import AffineTransform
from .. import utils


class VectorQuant(_VQBaseLayer):
	"""
	Vector quantization layer using straight-through estimation.

	Args:
		feature_size (int): feature dimension corresponding to the vectors
		num_codes (int): number of vectors in the codebook
		beta (float): commitment loss weighting
		sync_nu (float): sync loss weighting
		affine_lr (float): learning rate for affine transform
		affine_groups (int): number of affine parameter groups
		replace_freq (int): frequency to replace dead codes
		inplace_optimizer (Optimizer): optimizer for inplace codebook updates
		**kwargs: additional arguments for _VQBaseLayer

	Returns:
		Quantized vector z_q and return dict
	"""

	def __init__(
			self,
			feature_size: int,
			num_codes: int,
			beta: float = 0.97125,  # 0.9475 8.418
			sync_nu: float = 0.0,
			affine_lr: float = 0.0,
			affine_groups: int = 1,
			replace_freq: int = 0,
			inplace_optimizer: torch.optim.Optimizer = None,
			using_statistics: bool = False,
			use_learnable_std: bool = False,
			use_ema_update=False,
			decay=0.99,
			eps=1e-5,
			**kwargs,
	):

		super().__init__(feature_size, num_codes, **kwargs)
		self.loss_fn, self.dist_fn = get_dist_fns('euclidean')

		if beta < 0.0 or beta > 1.0:
			raise ValueError(f'beta must be in [0, 1] but got {beta}')

		self.beta = beta
		self.nu = sync_nu
		self.codebook = nn.Embedding(self.num_codes, self.feature_size)

		self.use_ema_update = use_ema_update
		if self.use_ema_update:
			self.register_buffer('cluster_size', torch.zeros(num_codes))
			self.register_buffer('embed_avg', self.codebook.weight.clone())
			self.decay = decay
			self.eps = eps
			self.codebook.requires_grad_ = False


		if inplace_optimizer is not None:
			if beta != 1.0:
				raise ValueError('inplace_optimizer can only be used with beta=1.0')
			print('using in inplace optimizer')
			self.inplace_codebook_optimizer = inplace_optimizer(self.codebook.parameters())

		if affine_lr > 0 or use_learnable_std:
			# defaults to using learnable affine parameters
			self.affine_transform = AffineTransform(
				self.code_vector_size,
				use_running_statistics=using_statistics,
				use_learnable_std=use_learnable_std,
				lr_scale=affine_lr,
				num_groups=affine_groups,
			)
		if replace_freq > 0:
			semivq.nn.utils.lru_replacement(self, rho=0.01, timeout=replace_freq)
		return

	def get_dynamic_info(self):
		if hasattr(self, 'affine_transform'):
			return self.affine_transform.get_dynamic_info()
		return None

	def straight_through_approximation(self, z, z_q):
		""" passed gradient from z_q to z """
		if self.nu > 0:
			z_q = z + (z_q - z).detach() + (self.nu * z_q) + (-self.nu * z_q).detach()
		else:
			z_q = z + (z_q - z).detach()
		return z_q

	def get_inner_layer(self):
		if not hasattr(self, 'affine_transform'):
			raise Exception('affine_transform is None')
		return self.affine_transform.get_inner_layer()

	def compute_loss(self, z_e, z_q):
		""" computes loss between z and z_q """
		if self.use_ema_update:
			return self.beta * self.loss_fn(z_e, z_q.detach())
		return ((1.0 - self.beta) * self.loss_fn(z_e, z_q.detach()) + \
				(self.beta) * self.loss_fn(z_e.detach(), z_q))

	def quantize(self, codebook, z):
		"""
		Quantizes the latent codes z with the codebook

		Args:
			codebook (Tensor): B x F
			z (Tensor): B x ... x F
		"""

		# reshape to (BHWG x F//G) and compute distance
		z_shape = z.shape[:-1]
		z_flat = z.view(z.size(0), -1, z.size(-1))
		if hasattr(self, 'affine_transform'):
			self.affine_transform.update_running_statistics(z_flat, codebook)
			codebook, alpha = self.affine_transform(codebook)

		with torch.no_grad():
			dist_out = self.dist_fn(
				tensor=z_flat,
				codebook=codebook,
				topk=self.topk,
				compute_chunk_size=self.cdist_chunk_size,
				half_precision=(z.is_cuda),
			)

			d = dist_out['d'].view(z_shape)
			q = dist_out['q'].view(z_shape).long()



		z_q = F.embedding(q, codebook)


		if self.training and hasattr(self, 'inplace_codebook_optimizer'):
			# update codebook inplace 
			inplace_loss = ((z_q - z.detach()) ** 2).mean()
			inplace_loss.backward(retain_graph=True)
			self.inplace_codebook_optimizer.step()
			self.inplace_codebook_optimizer.zero_grad()

			# forward pass again with the update codebook
			z_q = F.embedding(q, codebook)

		# NOTE to save compute, we assumed Q did not change.
		return z_q, d, q

	@torch.no_grad()
	def get_codebook(self):
		cb = self.codebook.weight
		if hasattr(self, 'affine_transform'):
		 	cb, alpha = self.affine_transform(cb)
		return cb
	

	@torch.no_grad()
	def get_alpha(self):
		if hasattr(self, 'affine_transform'):
			return self.affine_transform.get_alpha()
		return None

	def get_codebook_affine_params(self):
		if hasattr(self, 'affine_transform'):
			return self.affine_transform.get_affine_params()
		return None

	@with_codebook_normalization
	def forward(self, z):

		######
		## (1) formatting data by groups and invariant to dim
		######

		z = self.prepare_inputs(z, self.groups)
		if not self.enabled:
			z = self.to_original_format(z)
			return z, {}


		######
		## (2) quantize latent vector
		######

		z_q, d, q = self.quantize(self.codebook.weight, z)


		encodings = F.one_hot(q, num_classes=self.num_codes)
		e_mean = encodings.view(-1, self.num_codes).float().mean(0)
		perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-7)))
		active_ratio = q.unique().numel() / self.num_codes * 100

		to_return = {
			'z': z,  # each group input z_e
			'z_q': z_q,  # quantized output z_q
			'd': d,  # distance function for each group
			'q'	: q,				# codes
			'loss': self.compute_loss(z, z_q).mean(),
			'perplexity': perplexity,
			'active_ratio': active_ratio,
		}

		z_q = self.straight_through_approximation(z, z_q)
		z_q = self.to_original_format(z_q)
		
		if self.use_ema_update and self.training: 
			with torch.no_grad():
			# bhw * num_codes
				encodings = encodings.view(-1, self.num_codes)
				# bhw * feature
				z_flatten = z.reshape(-1, self.feature_size)
				self.cluster_size.data.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
				# (num_codes * bhw) * (bhw, feature) = (num * feature)
				self.embed_avg.data.mul_(self.decay).add_(encodings.transpose(0, 1).type(z_flatten.dtype) @ z_flatten, alpha=1 - self.decay)
				n = self.cluster_size.sum()
				smoothed_cluster_size = ((self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n)
				codebook_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
				self.codebook.weight.copy_(codebook_normalized)	


		return z_q, to_return

	def get_last_mean(self):
		if hasattr(self, 'affine_transform'):
			return self.affine_transform.get_last_mean()
		return None
	
	def get_codebook_entry(self, indices, shape):
		# get quantized latent vectors
		codebook = self.get_codebook()
		z_q = F.embedding(indices, codebook)
		# z_q = self.to_original_format(z_q)

		if shape is not None:
			z_q = z_q.view(shape)
			# reshape back to match original input shape
			z_q = z_q.permute(0, 3, 1, 2).contiguous()

		return z_q
	