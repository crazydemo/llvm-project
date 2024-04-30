// RUN: mlir-opt -split-input-file -transform-interpreter -cse  %s | FileCheck %s

func.func @decompose_1d_split(%arg0 : tensor<10xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  %0, %1 = tensor.split dim(0) strides([5]) %arg0
             : (tensor<10xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  return %0, %1 : tensor<5xf32>, tensor<5xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.decompose_split
    } : !transform.op<"func.func">
    transform.yield
  }
}
