/* DO NOT EDIT THIS FILE - it is machine generated */

/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.jni.generated.tensors

import org.platanios.tensorflow.jni.TensorFlow

object Basic {
  TensorFlow.load()

  @native def zerosLike(contextHandle: Long, x: Long): Long
  @native def onesLike(contextHandle: Long, x: Long): Long
  @native def fill(contextHandle: Long, dims: Long, value: Long): Long
  @native def rank(contextHandle: Long, input: Long): Long
  @native def size(contextHandle: Long, input: Long, out_type: Int): Long
  @native def shape(contextHandle: Long, input: Long, out_type: Int): Long
  @native def expandDims(contextHandle: Long, input: Long, dim: Long): Long
  @native def squeeze(contextHandle: Long, input: Long, squeeze_dims: Array[Long]): Long
  @native def pack(contextHandle: Long, values: Array[Long], axis: Long): Long
  @native def parallelConcat(contextHandle: Long, values: Array[Long], shape: Array[Long]): Long
  @native def unpack(contextHandle: Long, value: Long, num: Long, axis: Long): Array[Long]
  @native def concatV2(contextHandle: Long, values: Array[Long], axis: Long): Long
  @native def concatOffset(contextHandle: Long, concat_dim: Long, shape: Array[Long]): Array[Long]
  @native def split(contextHandle: Long, split_dim: Long, value: Long, num_split: Long): Array[Long]
  @native def splitV(contextHandle: Long, value: Long, size_splits: Long, split_dim: Long, num_split: Long): Array[Long]
  @native def tile(contextHandle: Long, input: Long, multiples: Long): Long
  @native def pad(contextHandle: Long, input: Long, paddings: Long): Long
  @native def padV2(contextHandle: Long, input: Long, paddings: Long, constant_values: Long): Long
  @native def mirrorPad(contextHandle: Long, input: Long, paddings: Long, mode: Array[Byte]): Long
  @native def reshape(contextHandle: Long, tensor: Long, shape: Long): Long
  @native def transpose(contextHandle: Long, x: Long, perm: Long): Long
  @native def conjugateTranspose(contextHandle: Long, x: Long, perm: Long): Long
  @native def invertPermutation(contextHandle: Long, x: Long): Long
  @native def reverseV2(contextHandle: Long, tensor: Long, axis: Long): Long
  @native def reverseSequence(contextHandle: Long, input: Long, seq_lengths: Long, seq_dim: Long, batch_dim: Long): Long
  @native def spaceToBatchND(contextHandle: Long, input: Long, block_shape: Long, paddings: Long): Long
  @native def batchToSpaceND(contextHandle: Long, input: Long, block_shape: Long, crops: Long): Long
  @native def spaceToDepth(contextHandle: Long, input: Long, block_size: Long, data_format: Array[Byte]): Long
  @native def depthToSpace(contextHandle: Long, input: Long, block_size: Long, data_format: Array[Byte]): Long
  @native def where(contextHandle: Long, input: Long): Long
  @native def unique(contextHandle: Long, x: Long, out_idx: Int): Array[Long]
  @native def uniqueWithCounts(contextHandle: Long, x: Long, out_idx: Int): Array[Long]
  @native def listDiff(contextHandle: Long, x: Long, y: Long, out_idx: Int): Array[Long]
  @native def gatherV2(contextHandle: Long, params: Long, indices: Long, axis: Long): Long
  @native def gatherNd(contextHandle: Long, params: Long, indices: Long): Long
  @native def scatterNd(contextHandle: Long, indices: Long, updates: Long, shape: Long): Long
  @native def slice(contextHandle: Long, input: Long, begin: Long, size: Long): Long
  @native def stridedSlice(contextHandle: Long, input: Long, begin: Long, end: Long, strides: Long, begin_mask: Long, end_mask: Long, ellipsis_mask: Long, new_axis_mask: Long, shrink_axis_mask: Long): Long
  @native def checkNumerics(contextHandle: Long, tensor: Long, message: Array[Byte]): Long
  @native def editDistance(contextHandle: Long, hypothesis_indices: Long, hypothesis_values: Long, hypothesis_shape: Long, truth_indices: Long, truth_values: Long, truth_shape: Long, normalize: Boolean): Long
  @native def oneHot(contextHandle: Long, indices: Long, depth: Long, on_value: Long, off_value: Long, axis: Long): Long
  @native def broadcastArgs(contextHandle: Long, s0: Long, s1: Long): Long
  @native def stopGradient(contextHandle: Long, input: Long): Long
  @native def preventGradient(contextHandle: Long, input: Long, message: Array[Byte]): Long
  @native def identity(contextHandle: Long, input: Long): Long
  @native def identityN(contextHandle: Long, input: Long): Array[Long]
  @native def scatterNdNonAliasingAdd(contextHandle: Long, input: Long, indices: Long, updates: Long): Long
  @native def quantizeAndDequantizeV3(contextHandle: Long, input: Long, input_min: Long, input_max: Long, num_bits: Long, signed_input: Boolean, range_given: Boolean): Long
  @native def quantizeV2(contextHandle: Long, input: Long, min_range: Long, max_range: Long, t: Int, mode: Array[Byte], round_mode: Array[Byte]): Array[Long]
  @native def dequantize(contextHandle: Long, input: Long, min_range: Long, max_range: Long, mode: Array[Byte]): Long
  @native def quantizedConcat(contextHandle: Long, concat_dim: Long, values: Array[Long], input_mins: Array[Long], input_maxes: Array[Long]): Array[Long]
  @native def quantizedReshape(contextHandle: Long, tensor: Long, shape: Long, input_min: Long, input_max: Long): Array[Long]
  @native def quantizedInstanceNorm(contextHandle: Long, x: Long, x_min: Long, x_max: Long, output_range_given: Boolean, given_y_min: Float, given_y_max: Float, variance_epsilon: Float, min_separation: Float): Array[Long]
  @native def fakeQuantWithMinMaxArgs(contextHandle: Long, inputs: Long, min: Float, max: Float, num_bits: Long, narrow_range: Boolean): Long
  @native def fakeQuantWithMinMaxVars(contextHandle: Long, inputs: Long, min: Long, max: Long, num_bits: Long, narrow_range: Boolean): Long
  @native def fakeQuantWithMinMaxVarsPerChannel(contextHandle: Long, inputs: Long, min: Long, max: Long, num_bits: Long, narrow_range: Boolean): Long
}
