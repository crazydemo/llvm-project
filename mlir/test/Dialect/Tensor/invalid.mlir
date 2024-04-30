// RUN: mlir-opt <%s -split-input-file -verify-diagnostics

// -----

func.func @concat_empty() {
  // expected-error@+1 {{requires at least one input}}
  %0 = tensor.concat dim(0) : () -> tensor<1x2x3xf32>
  return
}

// -----

func.func @split_rank_mismatch(%arg0: tensor<2x1xf32>) {
  // expected-error@+1 {{rank of splited results must match input rank}}
  %0, %1 = tensor.split dim(0) strides([1]) %arg0 : (tensor<2x1xf32>) -> (tensor<1xf32>, tensor<1xf32>)
  return
}

// -----

func.func @split_dim_out_of_range(%arg0: tensor<3xf32>) {
  // expected-error@+1 {{split dim must be less than the tensor rank}}
  %0, %1 = tensor.split dim(1) strides([2,2]) %arg0 : (tensor<3xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  return
}

// -----

func.func @split_static_shape_mismatch(%arg0: tensor<7x5xf32>) {
  // expected-error@+1 {{static split size mismatch along non-splited dimension 1}}
  %0, %1 = tensor.split dim(0) strides([4]) %arg0 : (tensor<7x5xf32>) -> (tensor<4x4xf32>, tensor<3x5xf32>)
  return
}

// -----
