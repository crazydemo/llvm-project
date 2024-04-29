// RUN: mlir-opt -split-input-file -transform-interpreter -cse  %s | FileCheck %s

func.func @decompose_1d_split(%arg0 : tensor<1xf32>,
                            %arg1 : tensor<2xf32>,
                            %arg2 : tensor<3xf32>,
                            %arg3: tensor<4xf32>) -> tensor<10xf32> {
  %0, %1, %2, %3 = tensor.split dim(0) strides([3]) %arg0
             : tensor<10xf32> -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<1xf32>) 
  return %0, %1, %2, %3 : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<1xf32>
}
