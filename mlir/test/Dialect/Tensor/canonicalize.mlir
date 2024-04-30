// RUN: mlir-opt %s -split-input-file -canonicalize="test-convergence" | FileCheck %s

// CHECK-LABEL: fold_concat
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x2x?xi32>
func.func @fold_concat(%arg0: tensor<1x2x?xi32>) -> (tensor<1x2x3xi32>, tensor<1x2x?xi32>) {
  %0 = tensor.concat dim(2) %arg0 : (tensor<1x2x?xi32>) -> tensor<1x2x3xi32>
  // CHECK-NEXT: %[[CAST:.*]] = tensor.cast %[[ARG0]] : tensor<1x2x?xi32> to tensor<1x2x3xi32>
  %1 = tensor.concat dim(2) %arg0 : (tensor<1x2x?xi32>) -> tensor<1x2x?xi32>
  // CHECK-NEXT: return %[[CAST]], %[[ARG0]] : tensor<1x2x3xi32>, tensor<1x2x?xi32>
  return %0, %1 : tensor<1x2x3xi32>, tensor<1x2x?xi32>
}

// -----

// CHECK-LABEL: fold_split
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x2x5xi32>
func.func @fold_split(%arg0: tensor<1x2x5xi32>) -> (tensor<1x2x5xi32>) {
  %0 = tensor.split dim(2) strides([5]) %arg0 : (tensor<1x2x5xi32>) -> (tensor<1x2x5xi32>)
  // CHECK-NEXT: return %[[ARG0]] : tensor<1x2x5xi32>
  return %0 : tensor<1x2x5xi32>
}
