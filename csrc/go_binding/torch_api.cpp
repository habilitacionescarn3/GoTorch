// torch_api.cpp — C++ implementation of the GoTorch Go→C++ bridge.
// Single compilation unit for the entire Go-callable C API.
//
// Build:
//   g++ -std=c++17 -c csrc/go_binding/torch_api.cpp \
//       -I./torch/csrc/api/include -I./build/include \
//       -o csrc/go_binding/torch_api.o

#include "torch_api.h"
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <cstring>
#include <vector>
#include <memory>
#include <iostream>

// ─── Internal helpers ─────────────────────────────────────────────────────────

static torch::Device to_device(int d) {
    if (d == DEVICE_CUDA) return torch::Device(torch::kCUDA);
    if (d == DEVICE_MPS)  return torch::Device(torch::kMPS);
    return torch::Device(torch::kCPU);
}

static torch::ScalarType to_dtype(int t) {
    switch (t) {
        case DTYPE_FLOAT32: return torch::kFloat32;
        case DTYPE_FLOAT64: return torch::kFloat64;
        case DTYPE_INT32:   return torch::kInt32;
        case DTYPE_INT64:   return torch::kInt64;
        case DTYPE_BOOL:    return torch::kBool;
        default:            return torch::kFloat32;
    }
}

static at::Reduction::Reduction to_reduction(int r) {
    switch (r) {
        case 0: return at::Reduction::None;
        case 2: return at::Reduction::Sum;
        default: return at::Reduction::Mean;
    }
}

// Tensor cast/wrap helpers
static inline torch::Tensor* T(Tensor t)  { return reinterpret_cast<torch::Tensor*>(t); }
static inline Tensor newT(const torch::Tensor& t) { return reinterpret_cast<Tensor>(new torch::Tensor(t)); }

// Helper: build a Tensor* array on heap from a vector (caller must free)
static Tensor* vec_to_arr(const std::vector<torch::Tensor>& v, int64_t* count) {
    *count = (int64_t)v.size();
    if (v.empty()) return nullptr;
    Tensor* arr = new Tensor[v.size()];
    for (size_t i = 0; i < v.size(); i++) arr[i] = newT(v[i]);
    return arr;
}

// Helper: params array → vector<Tensor>
static std::vector<torch::Tensor> arr_to_vec(Tensor* params, int64_t count) {
    std::vector<torch::Tensor> v;
    v.reserve(count);
    for (int64_t i = 0; i < count; i++) v.push_back(*T(params[i]));
    return v;
}

// ─── Tensor creation ──────────────────────────────────────────────────────────

Tensor gotorch_zeros(int64_t* s, int n, int dt, int dev) {
    return newT(torch::zeros({s,s+n}, torch::TensorOptions().dtype(to_dtype(dt)).device(to_device(dev))));
}
Tensor gotorch_ones(int64_t* s, int n, int dt, int dev) {
    return newT(torch::ones({s,s+n}, torch::TensorOptions().dtype(to_dtype(dt)).device(to_device(dev))));
}
Tensor gotorch_randn(int64_t* s, int n, int dt, int dev) {
    return newT(torch::randn({s,s+n}, torch::TensorOptions().dtype(to_dtype(dt)).device(to_device(dev))));
}
Tensor gotorch_rand(int64_t* s, int n, int dt, int dev) {
    return newT(torch::rand({s,s+n}, torch::TensorOptions().dtype(to_dtype(dt)).device(to_device(dev))));
}
Tensor gotorch_full(int64_t* s, int n, double fv, int dt, int dev) {
    return newT(torch::full({s,s+n}, fv, torch::TensorOptions().dtype(to_dtype(dt)).device(to_device(dev))));
}
Tensor gotorch_eye(int64_t n, int dt, int dev) {
    return newT(torch::eye(n, torch::TensorOptions().dtype(to_dtype(dt)).device(to_device(dev))));
}
Tensor gotorch_arange(double start, double end, double step, int dt, int dev) {
    return newT(torch::arange(start, end, step, torch::TensorOptions().dtype(to_dtype(dt)).device(to_device(dev))));
}
Tensor gotorch_linspace(double start, double end, int64_t steps, int dt, int dev) {
    return newT(torch::linspace(start, end, steps, torch::TensorOptions().dtype(to_dtype(dt)).device(to_device(dev))));
}
Tensor gotorch_from_data(double* data, int64_t* shape, int ndim, int dt, int dev) {
    int64_t numel = 1;
    for (int i = 0; i < ndim; i++) numel *= shape[i];
    auto t = torch::from_blob(data, {shape,shape+ndim}, torch::kFloat64).clone();
    if (dt != DTYPE_FLOAT64) t = t.to(to_dtype(dt));
    if (dev != DEVICE_CPU)   t = t.to(to_device(dev));
    return newT(t);
}

// ─── Tensor properties ────────────────────────────────────────────────────────

int64_t gotorch_tensor_ndim(Tensor t)  { return T(t)->dim(); }
int64_t gotorch_tensor_numel(Tensor t) { return T(t)->numel(); }
double  gotorch_tensor_item(Tensor t)  { return T(t)->item<double>(); }
double* gotorch_tensor_data_ptr(Tensor t) { return T(t)->data_ptr<double>(); }
void    gotorch_tensor_free(Tensor t)  { delete T(t); }
void    gotorch_tensor_print(Tensor t) { std::cout << *T(t) << std::endl; }
int     gotorch_tensor_requires_grad(Tensor t) { return T(t)->requires_grad() ? 1 : 0; }
void    gotorch_tensor_set_requires_grad(Tensor t, int v) { T(t)->requires_grad_(v != 0); }
Tensor  gotorch_tensor_to_device(Tensor t, int dev) { return newT(T(t)->to(to_device(dev))); }
Tensor  gotorch_tensor_to_dtype(Tensor t, int dt)   { return newT(T(t)->to(to_dtype(dt))); }

int64_t* gotorch_tensor_shape(Tensor t) {
    auto sz = T(t)->sizes();
    int64_t* out = new int64_t[sz.size()];
    for (size_t i = 0; i < sz.size(); i++) out[i] = sz[i];
    return out;
}
Tensor gotorch_tensor_grad(Tensor t) {
    if (!T(t)->grad().defined()) return nullptr;
    return newT(T(t)->grad());
}
void gotorch_tensor_zero_grad(Tensor t) {
    if (T(t)->grad().defined()) T(t)->grad().zero_();
}

// ─── Shape ops ────────────────────────────────────────────────────────────────

Tensor gotorch_reshape(Tensor t, int64_t* s, int n)   { return newT(T(t)->reshape({s,s+n})); }
Tensor gotorch_view(Tensor t, int64_t* s, int n)       { return newT(T(t)->view({s,s+n})); }
Tensor gotorch_flatten(Tensor t, int64_t a, int64_t b) { return newT(T(t)->flatten(a, b)); }
Tensor gotorch_unflatten(Tensor t, int64_t dim, int64_t* sizes, int n) {
    return newT(T(t)->unflatten(dim, {sizes, sizes+n}));
}
Tensor gotorch_transpose(Tensor t, int64_t d0, int64_t d1) { return newT(T(t)->transpose(d0, d1)); }
Tensor gotorch_permute(Tensor t, int64_t* dims, int n) { return newT(T(t)->permute({dims,dims+n})); }
Tensor gotorch_t(Tensor t)                { return newT(T(t)->t()); }
Tensor gotorch_squeeze(Tensor t)          { return newT(T(t)->squeeze()); }
Tensor gotorch_squeeze_dim(Tensor t, int64_t dim) { return newT(T(t)->squeeze(dim)); }
Tensor gotorch_unsqueeze(Tensor t, int64_t dim)   { return newT(T(t)->unsqueeze(dim)); }
Tensor gotorch_contiguous(Tensor t)       { return newT(T(t)->contiguous()); }
Tensor gotorch_detach(Tensor t)           { return newT(T(t)->detach()); }
Tensor gotorch_clone(Tensor t)            { return newT(T(t)->clone()); }
Tensor gotorch_cat(Tensor* ts, int64_t n, int64_t dim) { return newT(torch::cat(arr_to_vec(ts,n), dim)); }
Tensor gotorch_stack(Tensor* ts, int64_t n, int64_t dim) { return newT(torch::stack(arr_to_vec(ts,n), dim)); }
Tensor gotorch_slice(Tensor t, int64_t dim, int64_t s, int64_t e, int64_t step) {
    return newT(T(t)->slice(dim, s, e, step));
}
Tensor gotorch_index_select(Tensor t, int64_t dim, Tensor idx) {
    return newT(torch::index_select(*T(t), dim, *T(idx)));
}
Tensor gotorch_pixel_shuffle(Tensor t, int64_t r) { return newT(torch::pixel_shuffle(*T(t), r)); }

// ─── Arithmetic ───────────────────────────────────────────────────────────────

Tensor gotorch_add(Tensor a, Tensor b)    { return newT(*T(a) + *T(b)); }
Tensor gotorch_sub(Tensor a, Tensor b)    { return newT(*T(a) - *T(b)); }
Tensor gotorch_mul(Tensor a, Tensor b)    { return newT(*T(a) * *T(b)); }
Tensor gotorch_div(Tensor a, Tensor b)    { return newT(*T(a) / *T(b)); }
Tensor gotorch_matmul(Tensor a, Tensor b) { return newT(torch::matmul(*T(a), *T(b))); }
Tensor gotorch_mm(Tensor a, Tensor b)     { return newT(torch::mm(*T(a), *T(b))); }
Tensor gotorch_bmm(Tensor a, Tensor b)    { return newT(torch::bmm(*T(a), *T(b))); }
Tensor gotorch_dot(Tensor a, Tensor b)    { return newT(torch::dot(*T(a), *T(b))); }
Tensor gotorch_add_scalar(Tensor a, double s) { return newT(*T(a) + s); }
Tensor gotorch_mul_scalar(Tensor a, double s) { return newT(*T(a) * s); }
Tensor gotorch_pow(Tensor a, double e)    { return newT(torch::pow(*T(a), e)); }
Tensor gotorch_neg(Tensor t)              { return newT(-*T(t)); }
Tensor gotorch_abs(Tensor t)              { return newT(torch::abs(*T(t))); }
Tensor gotorch_exp(Tensor t)              { return newT(torch::exp(*T(t))); }
Tensor gotorch_log(Tensor t)              { return newT(torch::log(*T(t))); }
Tensor gotorch_log2(Tensor t)             { return newT(torch::log2(*T(t))); }
Tensor gotorch_log10(Tensor t)            { return newT(torch::log10(*T(t))); }
Tensor gotorch_sqrt(Tensor t)             { return newT(torch::sqrt(*T(t))); }
Tensor gotorch_clamp(Tensor t, double mn, double mx) { return newT(torch::clamp(*T(t), mn, mx)); }

// ─── Reduction ────────────────────────────────────────────────────────────────

Tensor gotorch_sum(Tensor t)              { return newT(T(t)->sum()); }
Tensor gotorch_sum_dim(Tensor t, int64_t d, int kd) { return newT(T(t)->sum(d, kd!=0)); }
Tensor gotorch_mean(Tensor t)             { return newT(T(t)->mean()); }
Tensor gotorch_mean_dim(Tensor t, int64_t d, int kd) { return newT(T(t)->mean(d, kd!=0)); }
Tensor gotorch_max(Tensor t)              { return newT(T(t)->max()); }
Tensor gotorch_min(Tensor t)              { return newT(T(t)->min()); }
Tensor gotorch_std(Tensor t)              { return newT(T(t)->std()); }
Tensor gotorch_var(Tensor t)              { return newT(T(t)->var()); }
Tensor gotorch_argmax(Tensor t, int64_t d, int kd) { return newT(T(t)->argmax(d, kd!=0)); }
Tensor gotorch_argmin(Tensor t, int64_t d, int kd) { return newT(T(t)->argmin(d, kd!=0)); }

// ─── Activations ──────────────────────────────────────────────────────────────

Tensor gotorch_relu(Tensor t)                  { return newT(torch::relu(*T(t))); }
Tensor gotorch_relu_(Tensor t)                 { torch::relu_(*T(t)); return t; }
Tensor gotorch_leaky_relu(Tensor t, double ns) { return newT(torch::leaky_relu(*T(t), ns)); }
Tensor gotorch_sigmoid(Tensor t)               { return newT(torch::sigmoid(*T(t))); }
Tensor gotorch_tanh(Tensor t)                  { return newT(torch::tanh(*T(t))); }
Tensor gotorch_softmax(Tensor t, int64_t dim)  { return newT(torch::softmax(*T(t), dim)); }
Tensor gotorch_log_softmax(Tensor t, int64_t d){ return newT(torch::log_softmax(*T(t), d)); }
Tensor gotorch_gelu(Tensor t)                  { return newT(torch::gelu(*T(t))); }
Tensor gotorch_silu(Tensor t)                  { return newT(torch::silu(*T(t))); }
Tensor gotorch_elu(Tensor t, double a)         { return newT(torch::elu(*T(t), a)); }
Tensor gotorch_selu(Tensor t)                  { return newT(torch::selu(*T(t))); }
Tensor gotorch_mish(Tensor t)                  { return newT(torch::mish(*T(t))); }
Tensor gotorch_hardswish(Tensor t)             { return newT(torch::hardswish(*T(t))); }
Tensor gotorch_hardsigmoid(Tensor t)           { return newT(torch::hardsigmoid(*T(t))); }

// ─── Loss ─────────────────────────────────────────────────────────────────────

Tensor gotorch_mse_loss(Tensor p, Tensor tgt, int r) {
    return newT(torch::mse_loss(*T(p), *T(tgt), to_reduction(r)));
}
Tensor gotorch_cross_entropy(Tensor p, Tensor tgt, int r) {
    return newT(torch::cross_entropy_loss(*T(p), *T(tgt), {}, to_reduction(r)));
}
Tensor gotorch_bce_loss(Tensor p, Tensor tgt, int r) {
    return newT(torch::binary_cross_entropy(*T(p), *T(tgt), {}, to_reduction(r)));
}
Tensor gotorch_bce_with_logits(Tensor p, Tensor tgt, int r) {
    return newT(torch::binary_cross_entropy_with_logits(*T(p), *T(tgt), {}, {}, to_reduction(r)));
}
Tensor gotorch_nll_loss(Tensor lp, Tensor tgt, int r) {
    return newT(torch::nll_loss(*T(lp), *T(tgt), {}, to_reduction(r)));
}
Tensor gotorch_l1_loss(Tensor p, Tensor tgt, int r) {
    return newT(torch::l1_loss(*T(p), *T(tgt), to_reduction(r)));
}
Tensor gotorch_huber_loss(Tensor p, Tensor tgt, double delta, int r) {
    return newT(torch::huber_loss(*T(p), *T(tgt), to_reduction(r), delta));
}

// ─── Autograd ─────────────────────────────────────────────────────────────────

void gotorch_backward(Tensor t)                     { T(t)->backward(); }
void gotorch_backward_with_grad(Tensor t, Tensor g) { T(t)->backward(*T(g)); }
void gotorch_set_grad_enabled(int e)                { torch::GradMode::set_enabled(e != 0); }
int  gotorch_is_grad_enabled(void)                  { return torch::GradMode::is_enabled() ? 1 : 0; }

// ─── nn: Linear ───────────────────────────────────────────────────────────────

using LinImpl = torch::nn::Linear;
Module gotorch_nn_linear_new(int64_t in, int64_t out, int bias) {
    return new LinImpl(torch::nn::LinearOptions(in, out).bias(bias!=0));
}
static LinImpl* Lin(Module m) { return reinterpret_cast<LinImpl*>(m); }
Tensor  gotorch_nn_linear_forward(Module m, Tensor x) { return newT((*Lin(m))->forward(*T(x))); }
Tensor  gotorch_nn_linear_weight(Module m)  { return newT((*Lin(m))->weight); }
Tensor  gotorch_nn_linear_bias(Module m)    { return newT((*Lin(m))->bias); }
void    gotorch_nn_linear_free(Module m)    { delete Lin(m); }
void    gotorch_nn_linear_train(Module m, int mode) { (*Lin(m))->train(mode!=0); }
Tensor* gotorch_nn_linear_parameters(Module m, int64_t* count) {
    auto p = (*Lin(m))->parameters();
    return vec_to_arr(p, count);
}

// ─── nn: Conv2d ───────────────────────────────────────────────────────────────

using Conv2dImpl = torch::nn::Conv2d;
Module gotorch_nn_conv2d_new(int64_t ic, int64_t oc, int64_t k, int64_t s, int64_t p, int bias) {
    return new Conv2dImpl(torch::nn::Conv2dOptions(ic,oc,k).stride(s).padding(p).bias(bias!=0));
}
Module gotorch_nn_conv2d_new_full(int64_t ic, int64_t oc, int64_t k, int64_t s, int64_t p, int64_t d, int64_t g, int bias) {
    return new Conv2dImpl(torch::nn::Conv2dOptions(ic,oc,k).stride(s).padding(p).dilation(d).groups(g).bias(bias!=0));
}
static Conv2dImpl* Conv(Module m) { return reinterpret_cast<Conv2dImpl*>(m); }
Tensor  gotorch_nn_conv2d_forward(Module m, Tensor x) { return newT((*Conv(m))->forward(*T(x))); }
void    gotorch_nn_conv2d_free(Module m)  { delete Conv(m); }
void    gotorch_nn_conv2d_train(Module m, int mode) { (*Conv(m))->train(mode!=0); }
Tensor* gotorch_nn_conv2d_parameters(Module m, int64_t* count) {
    return vec_to_arr((*Conv(m))->parameters(), count);
}

// Conv1d
using Conv1dImpl = torch::nn::Conv1d;
Module gotorch_nn_conv1d_new(int64_t ic, int64_t oc, int64_t k, int64_t s, int64_t p, int bias) {
    return new Conv1dImpl(torch::nn::Conv1dOptions(ic,oc,k).stride(s).padding(p).bias(bias!=0));
}
static Conv1dImpl* Conv1(Module m) { return reinterpret_cast<Conv1dImpl*>(m); }
Tensor gotorch_nn_conv1d_forward(Module m, Tensor x) { return newT((*Conv1(m))->forward(*T(x))); }
void   gotorch_nn_conv1d_free(Module m) { delete Conv1(m); }

// ConvTranspose2d
using CT2dImpl = torch::nn::ConvTranspose2d;
Module gotorch_nn_convtranspose2d_new(int64_t ic,int64_t oc,int64_t k,int64_t s,int64_t p,int64_t op,int bias){
    return new CT2dImpl(torch::nn::ConvTranspose2dOptions(ic,oc,k).stride(s).padding(p).output_padding(op).bias(bias!=0));
}
static CT2dImpl* CT2(Module m) { return reinterpret_cast<CT2dImpl*>(m); }
Tensor gotorch_nn_convtranspose2d_forward(Module m, Tensor x) { return newT((*CT2(m))->forward(*T(x))); }
void   gotorch_nn_convtranspose2d_free(Module m) { delete CT2(m); }

// ─── nn: Normalization ────────────────────────────────────────────────────────

using BN1Impl = torch::nn::BatchNorm1d;
Module gotorch_nn_batchnorm1d_new(int64_t n) { return new BN1Impl(n); }
static BN1Impl* BN1(Module m) { return reinterpret_cast<BN1Impl*>(m); }
Tensor gotorch_nn_batchnorm1d_forward(Module m, Tensor x, int training) {
    (*BN1(m))->train(training!=0);
    return newT((*BN1(m))->forward(*T(x)));
}
void gotorch_nn_batchnorm1d_free(Module m)          { delete BN1(m); }
void gotorch_nn_batchnorm1d_train(Module m, int mode) { (*BN1(m))->train(mode!=0); }

using BN2Impl = torch::nn::BatchNorm2d;
Module gotorch_nn_batchnorm2d_new(int64_t n) { return new BN2Impl(n); }
static BN2Impl* BN2(Module m) { return reinterpret_cast<BN2Impl*>(m); }
Tensor gotorch_nn_batchnorm2d_forward(Module m, Tensor x, int training) {
    (*BN2(m))->train(training!=0);
    return newT((*BN2(m))->forward(*T(x)));
}
void gotorch_nn_batchnorm2d_free(Module m)           { delete BN2(m); }
void gotorch_nn_batchnorm2d_train(Module m, int mode) { (*BN2(m))->train(mode!=0); }

using LNImpl = torch::nn::LayerNorm;
Module gotorch_nn_layernorm_new(int64_t* ns, int n, double eps) {
    return new LNImpl(torch::nn::LayerNormOptions({ns,ns+n}).eps(eps));
}
static LNImpl* LN(Module m) { return reinterpret_cast<LNImpl*>(m); }
Tensor gotorch_nn_layernorm_forward(Module m, Tensor x) { return newT((*LN(m))->forward(*T(x))); }
void   gotorch_nn_layernorm_free(Module m) { delete LN(m); }

using GNImpl = torch::nn::GroupNorm;
Module gotorch_nn_groupnorm_new(int64_t g, int64_t c, double eps) {
    return new GNImpl(torch::nn::GroupNormOptions(g, c).eps(eps));
}
static GNImpl* GN(Module m) { return reinterpret_cast<GNImpl*>(m); }
Tensor gotorch_nn_groupnorm_forward(Module m, Tensor x) { return newT((*GN(m))->forward(*T(x))); }
void   gotorch_nn_groupnorm_free(Module m) { delete GN(m); }

using IN2Impl = torch::nn::InstanceNorm2d;
Module gotorch_nn_instancenorm2d_new(int64_t n) { return new IN2Impl(n); }
static IN2Impl* IN2(Module m) { return reinterpret_cast<IN2Impl*>(m); }
Tensor gotorch_nn_instancenorm2d_forward(Module m, Tensor x, int training) {
    (*IN2(m))->train(training!=0);
    return newT((*IN2(m))->forward(*T(x)));
}
void gotorch_nn_instancenorm2d_free(Module m) { delete IN2(m); }

// ─── nn: Dropout ──────────────────────────────────────────────────────────────

Tensor gotorch_nn_dropout_forward(Tensor x, double p, int training) {
    auto opts = torch::nn::functional::DropoutFuncOptions().p(p).training(training!=0);
    return newT(torch::nn::functional::dropout(*T(x), opts));
}
Tensor gotorch_nn_dropout2d_forward(Tensor x, double p, int training) {
    auto opts = torch::nn::functional::Dropout2dFuncOptions().p(p).training(training!=0);
    return newT(torch::nn::functional::dropout2d(*T(x), opts));
}
Tensor gotorch_nn_alpha_dropout_forward(Tensor x, double p, int training) {
    auto opts = torch::nn::functional::AlphaDropoutFuncOptions().p(p).training(training!=0);
    return newT(torch::nn::functional::alpha_dropout(*T(x), opts));
}

// ─── nn: Pooling ──────────────────────────────────────────────────────────────

Tensor gotorch_nn_max_pool1d(Tensor x, int64_t k, int64_t s, int64_t p) {
    auto opts = torch::nn::functional::MaxPool1dFuncOptions(k).stride(s).padding(p);
    return newT(torch::nn::functional::max_pool1d(*T(x), opts));
}
Tensor gotorch_nn_max_pool2d(Tensor x, int64_t k, int64_t s, int64_t p, int64_t d, int cm) {
    auto opts = torch::nn::functional::MaxPool2dFuncOptions(k)
        .stride(s).padding(p).dilation(d).ceil_mode(cm!=0);
    return newT(torch::nn::functional::max_pool2d(*T(x), opts));
}
Tensor gotorch_nn_avg_pool2d(Tensor x, int64_t k, int64_t s, int64_t p, int cip) {
    auto opts = torch::nn::functional::AvgPool2dFuncOptions(k)
        .stride(s).padding(p).count_include_pad(cip!=0);
    return newT(torch::nn::functional::avg_pool2d(*T(x), opts));
}
Tensor gotorch_nn_adaptive_avg_pool2d(Tensor x, int64_t oh, int64_t ow) {
    return newT(torch::nn::functional::adaptive_avg_pool2d(*T(x), {oh, ow}));
}
Tensor gotorch_nn_adaptive_max_pool2d(Tensor x, int64_t oh, int64_t ow) {
    auto [out, idx] = torch::nn::functional::adaptive_max_pool2d_with_indices(*T(x), {oh, ow});
    return newT(out);
}

// ─── nn: Embedding ────────────────────────────────────────────────────────────

using EmbImpl = torch::nn::Embedding;
Module gotorch_nn_embedding_new(int64_t ne, int64_t ed, int64_t pi, int sparse) {
    auto opts = torch::nn::EmbeddingOptions(ne, ed).sparse(sparse!=0);
    if (pi >= 0) opts.padding_idx(pi);
    return new EmbImpl(opts);
}
static EmbImpl* Emb(Module m) { return reinterpret_cast<EmbImpl*>(m); }
Tensor  gotorch_nn_embedding_forward(Module m, Tensor idx) { return newT((*Emb(m))->forward(*T(idx))); }
Tensor  gotorch_nn_embedding_weight(Module m) { return newT((*Emb(m))->weight); }
void    gotorch_nn_embedding_free(Module m)   { delete Emb(m); }
Tensor* gotorch_nn_embedding_parameters(Module m, int64_t* count) {
    return vec_to_arr((*Emb(m))->parameters(), count);
}

using EmbBagImpl = torch::nn::EmbeddingBag;
Module gotorch_nn_embedding_bag_new(int64_t ne, int64_t ed, int mode) {
    torch::nn::EmbeddingBagMode m;
    if (mode == 0) m = torch::kSum;
    else if (mode == 2) m = torch::kMax;
    else m = torch::kMean;
    return new EmbBagImpl(torch::nn::EmbeddingBagOptions(ne, ed).mode(m));
}
static EmbBagImpl* EmbBag(Module m) { return reinterpret_cast<EmbBagImpl*>(m); }
Tensor gotorch_nn_embedding_bag_forward(Module m, Tensor input, Tensor offsets) {
    return newT((*EmbBag(m))->forward(*T(input), *T(offsets)));
}
Tensor gotorch_nn_embedding_bag_weight(Module m) { return newT((*EmbBag(m))->weight); }
void   gotorch_nn_embedding_bag_free(Module m)   { delete EmbBag(m); }

// ─── nn: LSTM ─────────────────────────────────────────────────────────────────

using LSTMImpl = torch::nn::LSTM;
Module gotorch_nn_lstm_new(int64_t is, int64_t hs, int64_t nl, int bias, int bf, double drop, int bidir) {
    auto opts = torch::nn::LSTMOptions(is, hs)
        .num_layers(nl).bias(bias!=0).batch_first(bf!=0)
        .dropout(drop).bidirectional(bidir!=0);
    return new LSTMImpl(opts);
}
static LSTMImpl* LSTM_(Module m) { return reinterpret_cast<LSTMImpl*>(m); }
void gotorch_nn_lstm_forward(Module m, Tensor x, Tensor h0, Tensor c0,
                              Tensor* out, Tensor* hn, Tensor* cn) {
    std::tuple<torch::Tensor, torch::Tensor> hx_init;
    if (h0 && c0) hx_init = {*T(h0), *T(c0)};
    auto [output, hx] = (*LSTM_(m))->forward(*T(x), hx_init);
    *out = newT(output);
    *hn  = newT(std::get<0>(hx));
    *cn  = newT(std::get<1>(hx));
}
void    gotorch_nn_lstm_train(Module m, int mode) { (*LSTM_(m))->train(mode!=0); }
void    gotorch_nn_lstm_free(Module m)             { delete LSTM_(m); }
Tensor* gotorch_nn_lstm_parameters(Module m, int64_t* count) {
    return vec_to_arr((*LSTM_(m))->parameters(), count);
}

// ─── nn: GRU ──────────────────────────────────────────────────────────────────

using GRUImpl = torch::nn::GRU;
Module gotorch_nn_gru_new(int64_t is, int64_t hs, int64_t nl, int bias, int bf, double drop, int bidir) {
    auto opts = torch::nn::GRUOptions(is, hs)
        .num_layers(nl).bias(bias!=0).batch_first(bf!=0)
        .dropout(drop).bidirectional(bidir!=0);
    return new GRUImpl(opts);
}
static GRUImpl* GRU_(Module m) { return reinterpret_cast<GRUImpl*>(m); }
void gotorch_nn_gru_forward(Module m, Tensor x, Tensor h0, Tensor* out, Tensor* hn) {
    c10::optional<torch::Tensor> h_init = h0 ? c10::optional<torch::Tensor>(*T(h0)) : c10::nullopt;
    auto [output, h_n] = (*GRU_(m))->forward(*T(x), h_init);
    *out = newT(output);
    *hn  = newT(h_n);
}
void    gotorch_nn_gru_train(Module m, int mode) { (*GRU_(m))->train(mode!=0); }
void    gotorch_nn_gru_free(Module m)             { delete GRU_(m); }
Tensor* gotorch_nn_gru_parameters(Module m, int64_t* count) {
    return vec_to_arr((*GRU_(m))->parameters(), count);
}

// ─── nn: MultiheadAttention ───────────────────────────────────────────────────

using MHAImpl = torch::nn::MultiheadAttention;
Module gotorch_nn_mha_new(int64_t ed, int64_t nh, double drop, int bias) {
    return new MHAImpl(torch::nn::MultiheadAttentionOptions(ed, nh).dropout(drop).bias(bias!=0));
}
static MHAImpl* MHA(Module m) { return reinterpret_cast<MHAImpl*>(m); }
void gotorch_nn_mha_forward(Module m, Tensor q, Tensor k, Tensor v, Tensor kpm,
                             Tensor* attn_out, Tensor* attn_w) {
    c10::optional<torch::Tensor> mask = kpm ? c10::optional<torch::Tensor>(*T(kpm)) : c10::nullopt;
    auto [out, weights] = (*MHA(m))->forward(*T(q), *T(k), *T(v), mask);
    *attn_out = newT(out);
    *attn_w   = newT(weights);
}
void    gotorch_nn_mha_train(Module m, int mode) { (*MHA(m))->train(mode!=0); }
void    gotorch_nn_mha_free(Module m)             { delete MHA(m); }
Tensor* gotorch_nn_mha_parameters(Module m, int64_t* count) {
    return vec_to_arr((*MHA(m))->parameters(), count);
}

// ─── nn: TransformerEncoderLayer ──────────────────────────────────────────────

using TELImpl = torch::nn::TransformerEncoderLayer;
Module gotorch_nn_transformer_enc_layer_new(int64_t dm, int64_t nh, int64_t dff, double drop) {
    return new TELImpl(torch::nn::TransformerEncoderLayerOptions(dm, nh)
        .dim_feedforward(dff).dropout(drop));
}
static TELImpl* TEL(Module m) { return reinterpret_cast<TELImpl*>(m); }
Tensor gotorch_nn_transformer_enc_layer_forward(Module m, Tensor src, Tensor mask) {
    c10::optional<torch::Tensor> m_opt = mask ? c10::optional<torch::Tensor>(*T(mask)) : c10::nullopt;
    return newT((*TEL(m))->forward(*T(src), m_opt));
}
void    gotorch_nn_transformer_enc_layer_train(Module m, int mode) { (*TEL(m))->train(mode!=0); }
void    gotorch_nn_transformer_enc_layer_free(Module m) { delete TEL(m); }
Tensor* gotorch_nn_transformer_enc_layer_parameters(Module m, int64_t* count) {
    return vec_to_arr((*TEL(m))->parameters(), count);
}

// ─── nn: TransformerEncoder ───────────────────────────────────────────────────

using TEImpl = torch::nn::TransformerEncoder;
Module gotorch_nn_transformer_encoder_new(Module layer, int64_t n) {
    // Clone the layer for the encoder stack
    auto* tel = TEL(layer);
    return new TEImpl(torch::nn::TransformerEncoderOptions(*tel, n));
}
static TEImpl* TE(Module m) { return reinterpret_cast<TEImpl*>(m); }
Tensor gotorch_nn_transformer_encoder_forward(Module m, Tensor src, Tensor mask) {
    c10::optional<torch::Tensor> m_opt = mask ? c10::optional<torch::Tensor>(*T(mask)) : c10::nullopt;
    return newT((*TE(m))->forward(*T(src), m_opt));
}
void    gotorch_nn_transformer_encoder_train(Module m, int mode) { (*TE(m))->train(mode!=0); }
void    gotorch_nn_transformer_encoder_free(Module m) { delete TE(m); }
Tensor* gotorch_nn_transformer_encoder_parameters(Module m, int64_t* count) {
    return vec_to_arr((*TE(m))->parameters(), count);
}

// ─── Optimizers ───────────────────────────────────────────────────────────────

Optimizer gotorch_optim_sgd_new(Tensor* ps, int64_t n, double lr, double mom, double wd, int nest) {
    auto opts = torch::optim::SGDOptions(lr).momentum(mom).weight_decay(wd).nesterov(nest!=0);
    return new torch::optim::SGD(arr_to_vec(ps,n), opts);
}
Optimizer gotorch_optim_adam_new(Tensor* ps, int64_t n, double lr, double b1, double b2, double eps, double wd) {
    auto opts = torch::optim::AdamOptions(lr).betas({b1,b2}).eps(eps).weight_decay(wd);
    return new torch::optim::Adam(arr_to_vec(ps,n), opts);
}
Optimizer gotorch_optim_adamw_new(Tensor* ps, int64_t n, double lr, double b1, double b2, double eps, double wd) {
    auto opts = torch::optim::AdamWOptions(lr).betas({b1,b2}).eps(eps).weight_decay(wd);
    return new torch::optim::AdamW(arr_to_vec(ps,n), opts);
}
Optimizer gotorch_optim_rmsprop_new(Tensor* ps, int64_t n, double lr, double alpha, double eps, double wd) {
    auto opts = torch::optim::RMSpropOptions(lr).alpha(alpha).eps(eps).weight_decay(wd);
    return new torch::optim::RMSprop(arr_to_vec(ps,n), opts);
}

static torch::optim::Optimizer* Opt(Optimizer o) { return reinterpret_cast<torch::optim::Optimizer*>(o); }
void   gotorch_optim_step(Optimizer o)      { Opt(o)->step(); }
void   gotorch_optim_zero_grad(Optimizer o) { Opt(o)->zero_grad(); }
void   gotorch_optim_free(Optimizer o)      { delete Opt(o); }

double gotorch_optim_get_lr(Optimizer o) {
    auto& pg = Opt(o)->param_groups()[0];
    return pg.options().get_lr();
}
void gotorch_optim_set_lr(Optimizer o, double lr) {
    for (auto& pg : Opt(o)->param_groups())
        pg.options().set_lr(lr);
}

// ─── CUDA ─────────────────────────────────────────────────────────────────────

int  gotorch_cuda_is_available(void)  { return torch::cuda::is_available() ? 1 : 0; }
int  gotorch_cuda_device_count(void)  { return (int)torch::cuda::device_count(); }
void gotorch_cuda_set_device(int id)  { torch::cuda::set_device(id); }
int  gotorch_cuda_current_device(void){ return (int)torch::cuda::current_device(); }
void gotorch_cuda_synchronize(void)   { torch::cuda::synchronize(); }

// ─── Serialization ────────────────────────────────────────────────────────────

void   gotorch_save_tensor(Tensor t, const char* p) { torch::save(*T(t), p); }
Tensor gotorch_load_tensor(const char* p)           { torch::Tensor t; torch::load(t, p); return newT(t); }

// ─── Functional ───────────────────────────────────────────────────────────────

Tensor gotorch_f_linear(Tensor x, Tensor w, Tensor b) {
    torch::Tensor bias = b ? *T(b) : torch::Tensor();
    return newT(torch::nn::functional::linear(*T(x), *T(w), bias));
}
Tensor gotorch_f_conv2d(Tensor x, Tensor w, Tensor b, int64_t s, int64_t p, int64_t d, int64_t g) {
    torch::Tensor bias = b ? *T(b) : torch::Tensor();
    auto opts = torch::nn::functional::Conv2dFuncOptions().stride(s).padding(p).dilation(d).groups(g);
    return newT(torch::nn::functional::conv2d(*T(x), *T(w), bias, opts));
}
Tensor gotorch_f_max_pool2d(Tensor x, int64_t k, int64_t s, int64_t p) {
    return newT(torch::nn::functional::max_pool2d(*T(x), torch::nn::functional::MaxPool2dFuncOptions(k).stride(s).padding(p)));
}
Tensor gotorch_f_avg_pool2d(Tensor x, int64_t k, int64_t s, int64_t p) {
    return newT(torch::nn::functional::avg_pool2d(*T(x), torch::nn::functional::AvgPool2dFuncOptions(k).stride(s).padding(p)));
}
Tensor gotorch_f_dropout(Tensor x, double p, int training) {
    return newT(torch::nn::functional::dropout(*T(x), torch::nn::functional::DropoutFuncOptions().p(p).training(training!=0)));
}
Tensor gotorch_f_embedding(Tensor w, Tensor idx) {
    return newT(torch::nn::functional::embedding(*T(idx), *T(w)));
}
Tensor gotorch_f_normalize(Tensor x, double p, int64_t dim) {
    return newT(torch::nn::functional::normalize(*T(x), torch::nn::functional::NormalizeFuncOptions().p(p).dim(dim)));
}
Tensor gotorch_f_interpolate(Tensor x, int64_t oh, int64_t ow, int mode) {
    // mode: 0=nearest, 1=bilinear, 2=bicubic
    torch::nn::functional::InterpolateFuncOptions opts;
    opts.size(std::vector<int64_t>{oh, ow});
    if (mode == 1) opts.mode(torch::kBilinear).align_corners(false);
    else if (mode == 2) opts.mode(torch::kBicubic).align_corners(false);
    else opts.mode(torch::kNearest);
    return newT(torch::nn::functional::interpolate(*T(x), opts));
}
