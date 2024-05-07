//===- SplitOpPatterns.cpp - Patterns related to tensor.split lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include <iostream>

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Decompose `tensor.split` into a chain of slice extract.
///
/// %split0, %split1 = tensor.split dim(1) strides([2]) %0 :
///         tensor<2x7xf32> -> (tensor<2x3xf32>, tensor<2x4xf32>)
///
/// Becomes
///
/// %split0 = tensor.extract_slice %0[0, 0][2, 3][1, 1]
/// %split1 = tensor.extract_slice %0[0, 3][2, 4][1, 1]
struct DecomposeTensorSplitOp : public OpRewritePattern<SplitOp> {
  using OpRewritePattern<SplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    Location loc = splitOp.getLoc();

    int64_t dim = splitOp.getDim();
    Value dimValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dim));

    int64_t rank = splitOp.getRank();
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    // Compute the partial sums for the slice offsets.
    AffineExpr sum = rewriter.getAffineDimExpr(0);
    SmallVector<AffineExpr> partialSums = {sum};
    SmallVector<OpFoldResult> offsetStrides = {rewriter.getIndexAttr(0)};
    for (auto [idx, input] :
         llvm::enumerate(splitOp.getResults().drop_back())) {
      sum = sum + rewriter.getAffineDimExpr(idx + 1);
      partialSums.push_back(sum);
      offsetStrides.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, input, dimValue));
    }
    auto partialSumMap = AffineMap::get(splitOp.getResults().size(), 0,
                                        partialSums, rewriter.getContext());
    SmallVector<OpFoldResult> dimOffsets =
        affine::makeComposedFoldedMultiResultAffineApply(
            rewriter, loc, partialSumMap, offsetStrides);

    SmallVector<Value> Results;
    Results.reserve(splitOp.getResults().size());
    for (auto [res, offset] :
         llvm::zip_equal(splitOp.getResults(), dimOffsets)) {
      offsets[dim] = offset;
      SmallVector<OpFoldResult> sizes =
          tensor::getMixedSizes(rewriter, loc, res);
      Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
          loc, splitOp.getInput(), offsets, sizes, strides);
      RankedTensorType slicedType =
          extractedSlice.getType().cast<RankedTensorType>();
      Results.push_back(extractedSlice);
    }
    rewriter.replaceOp(splitOp, Results);
    return success();
  }
};

} // namespace

void mlir::tensor::populateDecomposeTensorSplitPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DecomposeTensorSplitOp>(patterns.getContext());
}
