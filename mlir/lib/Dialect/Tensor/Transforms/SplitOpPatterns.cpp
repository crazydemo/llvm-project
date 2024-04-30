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

/// Decompose `tensor.split` into `tensor.empty` and a chain of slice inserts.
///
/// %split = tensor.split dim(1) strides([2]) %0 :
///         tensor<2x7xf32> -> (tensor<2x3xf32>, tensor<2x4xf32>)
///
/// Becomes
///
/// %empty0 = tensor.empty() : tensor<2x3xf32>
/// %empty1 = tensor.empty() : tensor<2x4xf32>
/// %split0 = tensor.extract_slice %0[0, 0][2, 3][1, 1]
/// %split1 = tensor.extract_slice %0[0, 3][2, 4][1, 1]
struct DecomposeTensorSplitOp : public OpRewritePattern<SplitOp> {
  using OpRewritePattern<SplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    std::cout << "split pattern decompose" << std::endl;
    Location loc = splitOp.getLoc();

    // Gather destination tensors.
    SmallVector<Value> dests;
    if (failed(tensor::getOrCreateDestinations(rewriter, loc, splitOp, dests)))
      return splitOp->emitOpError("failed to get destination tensors");

    SmallVector<tensor::EmptyOp> empties;
    for (auto dst : dests) {
      empties.push_back(dst.getDefiningOp<tensor::EmptyOp>());
    }

    int64_t dim = splitOp.getDim();
    Value dimValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dim));
    ArrayRef<int64_t> stride = splitOp.getDim();
    ArrayRef<int64_t> sections =
        splitOp.getSections(dim, stride, splitOp.getInputType());

    int64_t rank = splitOp.getRank();
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));

    // Compute the partial sums for the slice offsets.
    AffineExpr sum = rewriter.getAffineDimExpr(0);
    SmallVector<AffineExpr> partialSums = {sum};
    SmallVector<OpFoldResult> offsetStrides = {rewriter.getIndexAttr(0)};
    for (auto [idx, result] :
         llvm::enumerate(splitOp.getResults().drop_back())) {
      sum = sum + rewriter.getAffineDimExpr(idx + 1);
      partialSums.push_back(sum);
      offsetStrides.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, result, dimValue));
    }
    auto partialSumMap = AffineMap::get(splitOp.getResults().size(), 0,
                                        partialSums, rewriter.getContext());
    SmallVector<OpFoldResult> dimOffsets =
        affine::makeComposedFoldedMultiResultAffineApply(
            rewriter, loc, partialSumMap, offsetStrides);

    // Construct the chain of extract_slice ops into the destination.=
    // for (auto [result, offset] :
    //      llvm::zip_equal(splitOp.getResults(), dimOffsets)) {
    //   SmallVector<OpFoldResult> sizes =
    //       tensor::getMixedSizes(rewriter, loc, result);
    //   offsets[dim] = offset;
    //   rewriter.createOrFold<tensor::ExtractSliceOp>(
    //       loc, splitOp.getInputType(), result, offsets, sizes, strides);
    // }
    ReifiedRankedShapedTypeDims reifiedResultShapes;
    if (failed(reifyResultShapes(rewriter, splitOp, reifiedResultShapes))) {
      return rewriter.notifyMatchFailure(splitOp,
                                         "failed to reify result shapes");
    }
    assert(reifiedResultShapes.size() == splitOp.getResults().size() &&
           "expected same number of results");

    SmallVector<Value> Results;
    Results.reserve(splitOp.getResults().size());
    for (const auto &en : llvm::enumerate(splitOp.getResults())) {
      Value splitedResult = en.value();
      int64_t resultNumber = en.index();
      int64_t rank = splitOp.getRank();
      SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
      Results.push_back(rewriter.create<tensor::ExtractSliceOp>(
          loc, splitedResult, offsets, reifiedResultShapes[resultNumber],
          strides));
    }

    return success();
  }
};

} // namespace

void mlir::tensor::populateDecomposeTensorSplitPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DecomposeTensorSplitOp>(patterns.getContext());
}
