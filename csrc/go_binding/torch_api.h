// torch_api.h — C interface between Go (CGo) and GoTorch's C++ backend.
// Single header used by the root "gotorch" package.
// All functions use plain-C types so CGo can call them directly.
//
// Build (from GoTorch root):
//   mkdir build && cd build
//   cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF
//   cmake --build . --target torch torch_cpu c10 -j$(nproc)
//   cd ..
//   g++ -std=c++17 -c csrc/go_binding/torch_api.cpp \
//       -I./torch/csrc/api/include \
//       -I./build/include \
//       -I./csrc/go_binding \
//       -o csrc/go_binding/torch_api.o
//
// CGo flags (set in gotorch.go):
//   #cgo CFLAGS:  -I./torch/csrc/api/include -I./csrc/go_binding
//   #cgo LDFLAGS: -L./build/lib -Wl,-rpath,./build/lib -ltorch -ltorch_cpu -lc10

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

typedef void* Tensor;
typedef void* Module;
typedef void* Optimizer;

typedef enum { DEVICE_CPU=0, DEVICE_CUDA=1, DEVICE_MPS=2 } DeviceType;
typedef enum {
    DTYPE_FLOAT32=6, DTYPE_FLOAT64=7,
    DTYPE_INT32=3,   DTYPE_INT64=4, DTYPE_BOOL=11
} ScalarType;

// ─── Tensor creation ──────────────────────────────────────────────────────────
Tensor gotorch_zeros(int64_t* shape, int ndim, int dtype, int device);
Tensor gotorch_ones(int64_t* shape, int ndim, int dtype, int device);
Tensor gotorch_randn(int64_t* shape, int ndim, int dtype, int device);
Tensor gotorch_rand(int64_t* shape, int ndim, int dtype, int device);
Tensor gotorch_full(int64_t* shape, int ndim, double fill_value, int dtype, int device);
Tensor gotorch_eye(int64_t n, int dtype, int device);
Tensor gotorch_arange(double start, double end, double step, int dtype, int device);
Tensor gotorch_linspace(double start, double end, int64_t steps, int dtype, int device);
Tensor gotorch_from_data(double* data, int64_t* shape, int ndim, int dtype, int device);

// ─── Tensor properties ────────────────────────────────────────────────────────
int64_t  gotorch_tensor_ndim(Tensor t);
int64_t* gotorch_tensor_shape(Tensor t);
int64_t  gotorch_tensor_numel(Tensor t);
double   gotorch_tensor_item(Tensor t);
double*  gotorch_tensor_data_ptr(Tensor t);
void     gotorch_tensor_free(Tensor t);
void     gotorch_tensor_print(Tensor t);
int      gotorch_tensor_requires_grad(Tensor t);
void     gotorch_tensor_set_requires_grad(Tensor t, int val);
Tensor   gotorch_tensor_grad(Tensor t);
void     gotorch_tensor_zero_grad(Tensor t);
Tensor   gotorch_tensor_to_device(Tensor t, int device);
Tensor   gotorch_tensor_to_dtype(Tensor t, int dtype);

// ─── Shape ops ────────────────────────────────────────────────────────────────
Tensor gotorch_reshape(Tensor t, int64_t* shape, int ndim);
Tensor gotorch_view(Tensor t, int64_t* shape, int ndim);
Tensor gotorch_flatten(Tensor t, int64_t start_dim, int64_t end_dim);
Tensor gotorch_unflatten(Tensor t, int64_t dim, int64_t* sizes, int nsizes);
Tensor gotorch_transpose(Tensor t, int64_t dim0, int64_t dim1);
Tensor gotorch_permute(Tensor t, int64_t* dims, int ndims);
Tensor gotorch_t(Tensor t);
Tensor gotorch_squeeze(Tensor t);
Tensor gotorch_squeeze_dim(Tensor t, int64_t dim);
Tensor gotorch_unsqueeze(Tensor t, int64_t dim);
Tensor gotorch_contiguous(Tensor t);
Tensor gotorch_detach(Tensor t);
Tensor gotorch_clone(Tensor t);
Tensor gotorch_cat(Tensor* tensors, int64_t count, int64_t dim);
Tensor gotorch_stack(Tensor* tensors, int64_t count, int64_t dim);
Tensor gotorch_slice(Tensor t, int64_t dim, int64_t start, int64_t end, int64_t step);
Tensor gotorch_index_select(Tensor t, int64_t dim, Tensor indices);
Tensor gotorch_pixel_shuffle(Tensor t, int64_t upscale_factor);

// ─── Arithmetic ───────────────────────────────────────────────────────────────
Tensor gotorch_add(Tensor a, Tensor b);
Tensor gotorch_sub(Tensor a, Tensor b);
Tensor gotorch_mul(Tensor a, Tensor b);
Tensor gotorch_div(Tensor a, Tensor b);
Tensor gotorch_matmul(Tensor a, Tensor b);
Tensor gotorch_mm(Tensor a, Tensor b);
Tensor gotorch_bmm(Tensor a, Tensor b);
Tensor gotorch_dot(Tensor a, Tensor b);
Tensor gotorch_add_scalar(Tensor a, double scalar);
Tensor gotorch_mul_scalar(Tensor a, double scalar);
Tensor gotorch_pow(Tensor a, double exp);
Tensor gotorch_neg(Tensor t);
Tensor gotorch_abs(Tensor t);
Tensor gotorch_exp(Tensor t);
Tensor gotorch_log(Tensor t);
Tensor gotorch_log2(Tensor t);
Tensor gotorch_log10(Tensor t);
Tensor gotorch_sqrt(Tensor t);
Tensor gotorch_clamp(Tensor t, double min_val, double max_val);

// ─── Reduction ────────────────────────────────────────────────────────────────
Tensor gotorch_sum(Tensor t);
Tensor gotorch_sum_dim(Tensor t, int64_t dim, int keepdim);
Tensor gotorch_mean(Tensor t);
Tensor gotorch_mean_dim(Tensor t, int64_t dim, int keepdim);
Tensor gotorch_max(Tensor t);
Tensor gotorch_min(Tensor t);
Tensor gotorch_std(Tensor t);
Tensor gotorch_var(Tensor t);
Tensor gotorch_argmax(Tensor t, int64_t dim, int keepdim);
Tensor gotorch_argmin(Tensor t, int64_t dim, int keepdim);

// ─── Activations ──────────────────────────────────────────────────────────────
Tensor gotorch_relu(Tensor t);
Tensor gotorch_relu_(Tensor t);
Tensor gotorch_leaky_relu(Tensor t, double negative_slope);
Tensor gotorch_sigmoid(Tensor t);
Tensor gotorch_tanh(Tensor t);
Tensor gotorch_softmax(Tensor t, int64_t dim);
Tensor gotorch_log_softmax(Tensor t, int64_t dim);
Tensor gotorch_gelu(Tensor t);
Tensor gotorch_silu(Tensor t);
Tensor gotorch_elu(Tensor t, double alpha);
Tensor gotorch_selu(Tensor t);
Tensor gotorch_mish(Tensor t);
Tensor gotorch_hardswish(Tensor t);
Tensor gotorch_hardsigmoid(Tensor t);

// ─── Loss ─────────────────────────────────────────────────────────────────────
Tensor gotorch_mse_loss(Tensor pred, Tensor target, int reduction);
Tensor gotorch_cross_entropy(Tensor pred, Tensor target, int reduction);
Tensor gotorch_bce_loss(Tensor pred, Tensor target, int reduction);
Tensor gotorch_bce_with_logits(Tensor pred, Tensor target, int reduction);
Tensor gotorch_nll_loss(Tensor log_probs, Tensor target, int reduction);
Tensor gotorch_l1_loss(Tensor pred, Tensor target, int reduction);
Tensor gotorch_huber_loss(Tensor pred, Tensor target, double delta, int reduction);

// ─── Autograd ─────────────────────────────────────────────────────────────────
void gotorch_backward(Tensor t);
void gotorch_backward_with_grad(Tensor t, Tensor grad);
void gotorch_set_grad_enabled(int enabled);
int  gotorch_is_grad_enabled(void);

// ─── nn: Linear ───────────────────────────────────────────────────────────────
Module gotorch_nn_linear_new(int64_t in_features, int64_t out_features, int bias);
Tensor gotorch_nn_linear_forward(Module m, Tensor input);
Tensor gotorch_nn_linear_weight(Module m);
Tensor gotorch_nn_linear_bias(Module m);
void   gotorch_nn_linear_free(Module m);
void   gotorch_nn_linear_train(Module m, int mode);
Tensor* gotorch_nn_linear_parameters(Module m, int64_t* count);

// ─── nn: Conv2d ───────────────────────────────────────────────────────────────
Module gotorch_nn_conv2d_new(int64_t in_ch, int64_t out_ch, int64_t kernel,
                              int64_t stride, int64_t padding, int bias);
Module gotorch_nn_conv2d_new_full(int64_t in_ch, int64_t out_ch, int64_t kernel,
                                   int64_t stride, int64_t padding,
                                   int64_t dilation, int64_t groups, int bias);
Tensor  gotorch_nn_conv2d_forward(Module m, Tensor input);
Tensor* gotorch_nn_conv2d_parameters(Module m, int64_t* count);
void    gotorch_nn_conv2d_free(Module m);
void    gotorch_nn_conv2d_train(Module m, int mode);

// Conv1d
Module gotorch_nn_conv1d_new(int64_t in_ch, int64_t out_ch, int64_t kernel,
                              int64_t stride, int64_t padding, int bias);
Tensor gotorch_nn_conv1d_forward(Module m, Tensor input);
void   gotorch_nn_conv1d_free(Module m);

// ConvTranspose2d
Module gotorch_nn_convtranspose2d_new(int64_t in_ch, int64_t out_ch, int64_t kernel,
                                       int64_t stride, int64_t padding,
                                       int64_t output_padding, int bias);
Tensor gotorch_nn_convtranspose2d_forward(Module m, Tensor input);
void   gotorch_nn_convtranspose2d_free(Module m);

// ─── nn: Normalization ────────────────────────────────────────────────────────
Module gotorch_nn_batchnorm1d_new(int64_t num_features);
Tensor gotorch_nn_batchnorm1d_forward(Module m, Tensor input, int training);
void   gotorch_nn_batchnorm1d_free(Module m);
void   gotorch_nn_batchnorm1d_train(Module m, int mode);

Module gotorch_nn_batchnorm2d_new(int64_t num_features);
Tensor gotorch_nn_batchnorm2d_forward(Module m, Tensor input, int training);
void   gotorch_nn_batchnorm2d_free(Module m);
void   gotorch_nn_batchnorm2d_train(Module m, int mode);

Module gotorch_nn_layernorm_new(int64_t* normalized_shape, int ndim, double eps);
Tensor gotorch_nn_layernorm_forward(Module m, Tensor input);
void   gotorch_nn_layernorm_free(Module m);

Module gotorch_nn_groupnorm_new(int64_t num_groups, int64_t num_channels, double eps);
Tensor gotorch_nn_groupnorm_forward(Module m, Tensor input);
void   gotorch_nn_groupnorm_free(Module m);

Module gotorch_nn_instancenorm2d_new(int64_t num_features);
Tensor gotorch_nn_instancenorm2d_forward(Module m, Tensor input, int training);
void   gotorch_nn_instancenorm2d_free(Module m);

// ─── nn: Dropout ──────────────────────────────────────────────────────────────
Tensor gotorch_nn_dropout_forward(Tensor input, double p, int training);
Tensor gotorch_nn_dropout2d_forward(Tensor input, double p, int training);
Tensor gotorch_nn_alpha_dropout_forward(Tensor input, double p, int training);

// ─── nn: Pooling ──────────────────────────────────────────────────────────────
Tensor gotorch_nn_max_pool1d(Tensor input, int64_t kernel, int64_t stride, int64_t padding);
Tensor gotorch_nn_max_pool2d(Tensor input, int64_t kernel, int64_t stride,
                              int64_t padding, int64_t dilation, int ceil_mode);
Tensor gotorch_nn_avg_pool2d(Tensor input, int64_t kernel, int64_t stride,
                              int64_t padding, int count_include_pad);
Tensor gotorch_nn_adaptive_avg_pool2d(Tensor input, int64_t out_h, int64_t out_w);
Tensor gotorch_nn_adaptive_max_pool2d(Tensor input, int64_t out_h, int64_t out_w);

// ─── nn: Embedding ────────────────────────────────────────────────────────────
Module gotorch_nn_embedding_new(int64_t num_embeddings, int64_t embedding_dim,
                                 int64_t padding_idx, int sparse);
Tensor  gotorch_nn_embedding_forward(Module m, Tensor indices);
Tensor  gotorch_nn_embedding_weight(Module m);
void    gotorch_nn_embedding_free(Module m);
Tensor* gotorch_nn_embedding_parameters(Module m, int64_t* count);

Module gotorch_nn_embedding_bag_new(int64_t num_embeddings, int64_t embedding_dim, int mode);
Tensor gotorch_nn_embedding_bag_forward(Module m, Tensor input, Tensor offsets);
Tensor gotorch_nn_embedding_bag_weight(Module m);
void   gotorch_nn_embedding_bag_free(Module m);

// ─── nn: LSTM ─────────────────────────────────────────────────────────────────
Module gotorch_nn_lstm_new(int64_t input_size, int64_t hidden_size,
                            int64_t num_layers, int bias,
                            int batch_first, double dropout, int bidirectional);
void  gotorch_nn_lstm_forward(Module m, Tensor input, Tensor h0, Tensor c0,
                               Tensor* output, Tensor* hn, Tensor* cn);
void  gotorch_nn_lstm_train(Module m, int mode);
void  gotorch_nn_lstm_free(Module m);
Tensor* gotorch_nn_lstm_parameters(Module m, int64_t* count);

// ─── nn: GRU ──────────────────────────────────────────────────────────────────
Module gotorch_nn_gru_new(int64_t input_size, int64_t hidden_size,
                           int64_t num_layers, int bias,
                           int batch_first, double dropout, int bidirectional);
void  gotorch_nn_gru_forward(Module m, Tensor input, Tensor h0,
                              Tensor* output, Tensor* hn);
void  gotorch_nn_gru_train(Module m, int mode);
void  gotorch_nn_gru_free(Module m);
Tensor* gotorch_nn_gru_parameters(Module m, int64_t* count);

// ─── nn: MultiheadAttention ───────────────────────────────────────────────────
Module gotorch_nn_mha_new(int64_t embed_dim, int64_t num_heads,
                           double dropout, int bias);
void  gotorch_nn_mha_forward(Module m,
                              Tensor query, Tensor key, Tensor value,
                              Tensor key_padding_mask,
                              Tensor* attn_output, Tensor* attn_weights);
void  gotorch_nn_mha_train(Module m, int mode);
void  gotorch_nn_mha_free(Module m);
Tensor* gotorch_nn_mha_parameters(Module m, int64_t* count);

// ─── nn: TransformerEncoderLayer ──────────────────────────────────────────────
Module gotorch_nn_transformer_enc_layer_new(int64_t d_model, int64_t nhead,
                                             int64_t dim_feedforward, double dropout);
Tensor gotorch_nn_transformer_enc_layer_forward(Module m, Tensor src, Tensor src_mask);
void   gotorch_nn_transformer_enc_layer_train(Module m, int mode);
void   gotorch_nn_transformer_enc_layer_free(Module m);
Tensor* gotorch_nn_transformer_enc_layer_parameters(Module m, int64_t* count);

// ─── nn: TransformerEncoder ───────────────────────────────────────────────────
Module gotorch_nn_transformer_encoder_new(Module layer, int64_t num_layers);
Tensor gotorch_nn_transformer_encoder_forward(Module m, Tensor src, Tensor src_mask);
void   gotorch_nn_transformer_encoder_train(Module m, int mode);
void   gotorch_nn_transformer_encoder_free(Module m);
Tensor* gotorch_nn_transformer_encoder_parameters(Module m, int64_t* count);

// ─── Optimizers ───────────────────────────────────────────────────────────────
Optimizer gotorch_optim_sgd_new(Tensor* params, int64_t count,
                                 double lr, double momentum,
                                 double weight_decay, int nesterov);
Optimizer gotorch_optim_adam_new(Tensor* params, int64_t count,
                                  double lr, double beta1, double beta2,
                                  double eps, double weight_decay);
Optimizer gotorch_optim_adamw_new(Tensor* params, int64_t count,
                                   double lr, double beta1, double beta2,
                                   double eps, double weight_decay);
Optimizer gotorch_optim_rmsprop_new(Tensor* params, int64_t count,
                                     double lr, double alpha, double eps,
                                     double weight_decay);
void      gotorch_optim_step(Optimizer opt);
void      gotorch_optim_zero_grad(Optimizer opt);
void      gotorch_optim_free(Optimizer opt);
double    gotorch_optim_get_lr(Optimizer opt);
void      gotorch_optim_set_lr(Optimizer opt, double lr);

// ─── CUDA ─────────────────────────────────────────────────────────────────────
int  gotorch_cuda_is_available(void);
int  gotorch_cuda_device_count(void);
void gotorch_cuda_set_device(int device_id);
int  gotorch_cuda_current_device(void);
void gotorch_cuda_synchronize(void);

// ─── Serialization ────────────────────────────────────────────────────────────
void   gotorch_save_tensor(Tensor t, const char* path);
Tensor gotorch_load_tensor(const char* path);

// ─── Functional ───────────────────────────────────────────────────────────────
Tensor gotorch_f_linear(Tensor input, Tensor weight, Tensor bias);
Tensor gotorch_f_conv2d(Tensor input, Tensor weight, Tensor bias,
                         int64_t stride, int64_t padding,
                         int64_t dilation, int64_t groups);
Tensor gotorch_f_max_pool2d(Tensor input, int64_t kernel, int64_t stride, int64_t padding);
Tensor gotorch_f_avg_pool2d(Tensor input, int64_t kernel, int64_t stride, int64_t padding);
Tensor gotorch_f_dropout(Tensor input, double p, int training);
Tensor gotorch_f_embedding(Tensor weight, Tensor indices);
Tensor gotorch_f_normalize(Tensor input, double p, int64_t dim);
Tensor gotorch_f_interpolate(Tensor input, int64_t out_h, int64_t out_w, int mode);

#ifdef __cplusplus
}
#endif
