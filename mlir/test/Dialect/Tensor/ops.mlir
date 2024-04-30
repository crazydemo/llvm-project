 // RUN: mlir-opt --split-input-file %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @cast(
func.func @cast(%arg0: tensor<*xf32>, %arg1 : tensor<4x4xf32>, %arg2: tensor<?x?xf32>) {
  // CHECK: tensor.cast %{{.*}} : tensor<*xf32> to tensor<?x?xf32>
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  // CHECK: tensor.cast %{{.*}} : tensor<4x4xf32> to tensor<*xf32>
  %1 = tensor.cast %arg1 : tensor<4x4xf32> to tensor<*xf32>
  // CHECK: tensor.cast %{{.*}} : tensor<?x?xf32> to tensor<4x?xf32>
  %2 = tensor.cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>
  // CHECK: tensor.cast %{{.*}} : tensor<4x?xf32> to tensor<?x?xf32>
  %3 = tensor.cast %2 : tensor<4x?xf32> to tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @concat(
func.func @concat(%arg0: tensor<4x7x3xf32>, %arg1 : tensor<4x4x3xf32>, %arg2: tensor<?x?x?xf32>) {
  // CHECK: tensor.concat dim(0) %{{.*}} : (tensor<4x7x3xf32>) -> tensor<4x7x3xf32>
  %0 = tensor.concat dim(0) %arg0 : (tensor<4x7x3xf32>) -> tensor<4x7x3xf32>
  // CHECK: tensor.concat dim(1) %{{.*}} : (tensor<4x7x3xf32>, tensor<4x4x3xf32>) -> tensor<4x11x3xf32>
  %1 = tensor.concat dim(1) %arg0, %arg1 : (tensor<4x7x3xf32>, tensor<4x4x3xf32>) -> tensor<4x11x3xf32>
  // CHECK: tensor.concat dim(2) %{{.*}} : (tensor<4x7x3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = tensor.concat dim(2) %arg0, %arg2 : (tensor<4x7x3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK: tensor.concat dim(1) %{{.*}} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x10x?xf32>
  %3 = tensor.concat dim(1) %arg2, %arg2 : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x10x?xf32>
  // CHECK: tensor.concat dim(1) %{{.*}} : (tensor<?x?x?xf32>, tensor<4x4x3xf32>, tensor<4x7x3xf32>) -> tensor<4x?x3xf32>
  %4 = tensor.concat dim(1) %arg2, %arg1, %arg0 : (tensor<?x?x?xf32>, tensor<4x4x3xf32>, tensor<4x7x3xf32>) -> tensor<4x?x3xf32>
  return
}

// -----

// CHECK-LABEL: func @split(
func.func @split(%arg0: tensor<4x128x68xf32>) {
  // CHECK: tensor.split dim(0) strides([2]) %{{.*}} : (tensor<4x128x68xf32>) -> (tensor<2x128x68xf32>, tensor<2x128x68xf32>)
  %0, %1 = tensor.split dim(0) strides([2]) %arg0 : (tensor<4x128x68xf32>) -> (tensor<2x128x68xf32>, tensor<2x128x68xf32>)
  // CHECK: tensor.split dim(1) strides([64]) %{{.*}} : (tensor<4x128x68xf32>) -> (tensor<4x64x68xf32>, tensor<4x64x68xf32>)
  %2, %3 = tensor.split dim(1) strides([64]) %arg0 : (tensor<4x128x68xf32>) -> (tensor<4x64x68xf32>, tensor<4x64x68xf32>)
  // CHECK: tensor.split dim(1) strides([64, 35, 29]) %{{.*}} : (tensor<4x128x68xf32>) -> (tensor<4x64x68xf32>, tensor<4x35x68xf32>, tensor<4x29x68xf32>)
  %4, %5, %6 = tensor.split dim(1) strides([64, 35, 29]) %arg0 : (tensor<4x128x68xf32>) -> (tensor<4x64x68xf32>, tensor<4x35x68xf32>, tensor<4x29x68xf32>)
  // CHECK: tensor.split dim(2) strides([64]) %{{.*}} : (tensor<4x128x68xf32>) -> (tensor<4x128x64xf32>, tensor<4x128x4xf32>)
  %7, %8 = tensor.split dim(2) strides([64]) %arg0 : (tensor<4x128x68xf32>) -> (tensor<4x128x64xf32>, tensor<4x128x4xf32>)
  return
}
