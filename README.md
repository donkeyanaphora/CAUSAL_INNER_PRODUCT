
# Causal-Basis Steering for GPT-2

This repo explores **linear concept representations** and **activation steering** in GPT-2 using the framework from:

> Kiho Park, Yo Joong Choe, Victor Veitch.
> *The Linear Representation Hypothesis and the Geometry of Large Language Models*. ICML 2024. 

This repo is a small implementation of their **causal inner product** and **intervention** ideas for steering GPT-2’s outputs along a learned concept directions.

## Core idea
Given a causal language model with:

* **Hidden states** $\lambda(x) \in \mathbb{R}^d$ (last-layer token representations)
* **Unembedding (lm head) weights** $\gamma(y) \in \mathbb{R}^d$ for each token y

the paper shows that one can define a **causal inner product** and a corresponding **change of basis** that better respects semantic structure e.g. causally separable concepts will be orthogonal in this new basis. In their implementation:

1. **Estimate covariance of the unembedding matrix**

   ```python
   gamma = model.lm_head.weight.detach()    # (V, d)
   gamma_bar = gamma.mean(dim=0)
   centered_gamma = gamma - gamma_bar

   cov_gamma = centered_gamma.T @ centered_gamma / gamma.size(0)
   eigvals, eigvecs = torch.linalg.eigh(cov_gamma)

   inv_sqrt_cov_gamma = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
   sqrt_cov_gamma = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
   ```

2. **Define a causal basis** (matching the invariance transform in Eq. (3.1) of the paper): 

   Transformed unembedding (our **causal head**):
   
     ```python
     causal_lm_head = gamma @ inv_sqrt_cov_gamma # g(y) = gamma(y) @ A,  A = Cov(gamma)^(-1/2)
     ```
   Transformed hidden states:
   
     ```python
     l_causal = lambda_x @ sqrt_cov_gamma # l(x) = λ(x) A^{-1}, A^{-1} = Cov(gamma)^(+1/2)
     ```
   With these two transforms, logits are preserved: 
   
$$
\lambda(x)^\top \gamma(y) = l(x)^\top g(y)
$$

4. **Learn a concept direction in this causal basis**

   Script trains a simple logistic regression probe to separate two sets of contexts (e.g. male vs female):

   ```python
   X = torch.cat([m_emb, f_emb], dim=0) @ sqrt_cov_gamma  # embeddings in causal basis
   y = torch.cat([torch.ones(len(m_emb)), torch.zeros(len(f_emb))])

   clf = LogisticRegression(max_iter=1000, penalty=None, solver="lbfgs")
   clf.fit(X.cpu().numpy(), y.cpu().numpy())

   concept_dir = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)
   steer_dir_causal = concept_dir / concept_dir.norm()
   ```

5. **Steer the model in the causal basis**

   For a given prompt:

   ```python
   lambda_x = outputs.hidden_states[-1][:, last_token_idx, :]
   l_causal = lambda_x @ sqrt_cov_gamma
   l_steered = l_causal + alpha * steer_dir_causal
   causal_logit = l_steered @ causal_lm_head.T
   ```

   This mirrors the paper’s **intervention representation** $\lambda_{W,\alpha}(x) = \lambda(x) + \alpha ,\bar\lambda_W$ (Eq. (4.2)), but implemented in the unified causal basis where embedding and unembedding representations are aligned. 


## `SteerableGPT2` wrapper

To make steering convenient, we wrap GPT-2 in a small subclass:

```python
class SteerableGPT2(GPT2LMHeadModel):
    def __init__(self, base_model, lm_head_g, sqrt_cov_gamma, concept_dir, alpha: float = 0.0):
        super().__init__(base_model.config)
        # reuse base model's transformer + original head
        self.transformer = base_model.transformer
        self.lm_head= base_model.lm_head

        # g(y) = gamma(y) @ A where A = Cov(gamma)^(-1/2) 
        self.register_buffer("lm_head_g", lm_head_g)

        # A_inv = sqrt_cov_gamma = Cov(gamma)^(+1/2), used to map lambda -> l_causal
        self.register_buffer("sqrt_cov_gamma", sqrt_cov_gamma)

        # steering direction
        self.register_buffer("concept_dir", concept_dir)

        self.alpha = alpha
    
    def forward(self, *args, **kwargs):
        # get all hidden states so we can grab the last layer
        outputs = super().forward(*args, output_hidden_states=True, **kwargs)
        lambda_all = outputs.hidden_states[-1]   # shape: (batch, seq, d_model)

        # change basis -> steer -> compute logits
        l_causal = lambda_all @ self.sqrt_cov_gamma

        # steer only the last token
        l_causal[:, -1, :] = l_causal[:, -1, :] + self.alpha * self.concept_dir

        # logits = (l(x) + alpha * concept_dir).T @ g(y)
        outputs.logits = l_causal @ self.lm_head_g.T
        
        return outputs
```

## Dependencies

* Python 3.x
* [PyTorch](https://pytorch.org/)
* [transformers](https://huggingface.co/docs/transformers/)
* [scikit-learn](https://scikit-learn.org/)

Install (example):

```bash
pip install torch transformers scikit-learn tabulate
```

## Usage
- [View on GitHub](https://github.com/donkeyanaphora/CAUSAL_INNER_PRODUCT/blob/main/concept_steer.ipynb)
<!-- - [View on nbviewer](https://nbviewer.org/github/donkeyanaphora/CAUSAL_INNER_PRODUCT/blob/main/concept_steer.ipynb)
- [Open In Colab](https://colab.research.google.com/github/donkeyanaphora/CAUSAL_INNER_PRODUCT/blob/main/concept_steer.ipynb) -->

## Citing

If you use this code or its ideas in academic work, please cite the original paper:

```bibtex
@inproceedings{park2024linear,
  title     = {The Linear Representation Hypothesis and the Geometry of Large Language Models},
  author    = {Park, Kiho and Choe, Yo Joong and Veitch, Victor},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024}
}
```
