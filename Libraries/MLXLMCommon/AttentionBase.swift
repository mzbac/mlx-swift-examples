import Foundation
import MLX
import MLXFast

/// Scaled dot-product attention that works with both regular and quantized KV cache
/// Following the mlx-lm pattern from models/base.py
public func scaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    
    if let quantizedCache = cache as? QuantizedKVCache {
        guard let (qKeys, qValues) = quantizedCache.getQuantizedData() else {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask
            )
        }
        
        return quantizedScaledDotProductAttention(
            queries: queries,
            qKeys: qKeys,
            qValues: qValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedCache.groupSize,
            bits: quantizedCache.bits
        )
    } else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
}

/// Quantized scaled dot-product attention
/// Based on mlx-lm's quantized_scaled_dot_product_attention
private func quantizedScaledDotProductAttention(
    queries: MLXArray,
    qKeys: (MLXArray, MLXArray, MLXArray),
    qValues: (MLXArray, MLXArray, MLXArray),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    groupSize: Int = 64,
    bits: Int = 8
) -> MLXArray {
    let B = queries.shape[0]
    let nQHeads = queries.shape[1]
    let L = queries.shape[2]
    let D = queries.shape[3]
    let nKvHeads = qKeys.0.shape[1]
    let nRepeats = nQHeads / nKvHeads
    
    var queries = queries * scale
    var qKeys = qKeys
    var qValues = qValues
    
    // Handle grouped-query attention (GQA)
    if nRepeats > 1 {
        queries = queries.reshaped(B, nKvHeads, nRepeats, L, D)
        // Expand quantized keys/values for broadcasting
        qKeys = (
            qKeys.0.expandedDimensions(axis: 2),
            qKeys.1.expandedDimensions(axis: 2),
            qKeys.2.expandedDimensions(axis: 2)
        )
        qValues = (
            qValues.0.expandedDimensions(axis: 2),
            qValues.1.expandedDimensions(axis: 2),
            qValues.2.expandedDimensions(axis: 2)
        )
    }
    
    // First quantizedMatmul: queries @ keys^T
    var scores = MLX.quantizedMatmul(
        queries,
        qKeys.0,
        scales: qKeys.1,
        biases: qKeys.2,
        transpose: true,
        groupSize: groupSize,
        bits: bits
    )
    
    // Apply mask
    switch mask {
    case .none:
        break
    case .causal:
        // Create causal mask
        let qL = scores.shape[scores.shape.count - 2]
        let kL = scores.shape[scores.shape.count - 1]
        let qIndices = MLXArray(Int32(kL - qL) ..< Int32(kL))
        let kIndices = MLXArray(0 ..< Int32(kL))
        let causalMask = qIndices[0..., .newAxis] .>= kIndices[.newAxis, 0...]
        let minValue = MLXArray(scores.dtype.finfo!.min, dtype: scores.dtype)
        scores = MLX.where(causalMask, scores, minValue)
    case .array(let maskArray):
        if maskArray.dtype == .bool {
            let minValue = MLXArray(scores.dtype.finfo!.min, dtype: scores.dtype)
            scores = MLX.where(maskArray, scores, minValue)
        } else {
            scores = scores + maskArray
        }
    case .arrays:
        fatalError("Multiple mask arrays (.arrays) not supported for quantized attention")
    }
    
    // Apply softmax
    let probs = softmax(scores, axis: -1, precise: true)
    
    // Second quantizedMatmul: probs @ values
    var output = MLX.quantizedMatmul(
        probs,
        qValues.0,
        scales: qValues.1,
        biases: qValues.2,
        transpose: false,
        groupSize: groupSize,
        bits: bits
    )
    
    // Reshape output back if we expanded for grouped-query attention
    if nRepeats > 1 {
        output = output.reshaped(B, nQHeads, L, D)
    }
    
    return output
}