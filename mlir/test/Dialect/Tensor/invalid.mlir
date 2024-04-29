// RUN: mlir-opt <%s -split-input-file -verify-diagnostics

// -----

func.func @concat_empty() {
  // expected-error@+1 {{requires at least one input}}
  %0 = tensor.concat dim(0) : () -> tensor<1x2x3xf32>
  return
}

// -----

func.func @split_empty() {
  // expected-error@+1 {{requires at least one input}}
  %results = tensor.split dim(0) strides([2]) : () -> tensor<1x2x3xf32>
  return
}

// -----

func.func @split_rank_mismatch(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) {
  // expected-error@+1 {{rank of splited inputs must match result rank}}
  %0, 1% = tensor.split dim(0) strides([1]) %arg0 : tensor<2x1xf32> -> (tensor<1xf32>, tensor<1xf32>)
  return
}

// -----

func.func @split_dim_out_of_range(%arg0: tensor<3xf32>) {
  // expected-error@+1 {{split dim must be less than the tensor rank}}
  %0, %1 = tensor.split dim(1) strides([2]) %arg0 : tensor<3xf32> -> (tensor<2xf32>, tensor<1xf32>)
  return
}

// -----

func.func @split_static_shape_mismatch(%arg0: tensor<3xf32>) {
  // expected-error@+1 {{result type 'tensor<7xf32>'does not match inferred shape 'tensor<6xf32>' static sizes}}
  %0, %1 = tensor.split dim(0) strides([2]) %arg0 : tensor<7xf32> -> (tensor<3xf32>, tensor<3xf32>)
  return
}

// -----
