import Foundation
import MLX

/// Extension to KVCacheSimple to add quantization conversion capability
extension KVCacheSimple {
    /// Convert this cache to a quantized cache
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        let quantizedCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quantizedCache.offset = self.offset
        if let keys = self.keys, let values = self.values {
            // Store quantized versions of existing data
            quantizedCache.keys = MLX.quantized(keys, groupSize: groupSize, bits: bits)
            quantizedCache.values = MLX.quantized(values, groupSize: groupSize, bits: bits)
        }
        return quantizedCache
    }
}

/// Quantized KV cache implementation that stores keys and values in quantized format
/// Based on mlx-lm's QuantizedKVCache
public class QuantizedKVCache: KVCache, Evaluatable, CustomDebugStringConvertible {
    // Store quantized data as tuples (data, scales, biases)
    var keys: (MLXArray, MLXArray, MLXArray)?
    var values: (MLXArray, MLXArray, MLXArray)?
    
    public var offset = 0
    var step = 256
    public let groupSize: Int
    public let bits: Int
    public let maxSize: Int? = nil
    
    public init(groupSize: Int = 64, bits: Int = 8) {
        self.groupSize = groupSize
        self.bits = bits
    }
    
    public func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2])
        }
        if let values = values {
            arrays.append(contentsOf: [values.0, values.1, values.2])
        }
        return arrays
    }
    
    /// Update the cache with new keys/values
    /// For compatibility with KVCache protocol, returns dummy arrays
    /// The actual quantized data is stored internally and accessed via getQuantizedData()
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        let nKvHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = self.offset
        
        // Check if we need to allocate or expand the cache
        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(2) {
            let elPerInt = 32 / self.bits  // elements per int32
            let newSteps = ((self.step + numSteps - 1) / self.step) * self.step
            let shape = [B, nKvHeads, newSteps]
            
            func initQuant(dim: Int) -> (MLXArray, MLXArray, MLXArray) {
                return (
                    MLXArray.zeros(shape + [dim / elPerInt], dtype: .uint32),
                    MLXArray.zeros(shape + [dim / self.groupSize], dtype: keys.dtype),
                    MLXArray.zeros(shape + [dim / self.groupSize], dtype: keys.dtype)
                )
            }
            
            func expandQuant(_ x: (MLXArray, MLXArray, MLXArray)) -> (MLXArray, MLXArray, MLXArray) {
                let newData = MLXArray.zeros([B, nKvHeads, newSteps, x.0.dim(3)], dtype: x.0.dtype)
                let newScales = MLXArray.zeros([B, nKvHeads, newSteps, x.1.dim(3)], dtype: x.1.dtype)
                let newBiases = MLXArray.zeros([B, nKvHeads, newSteps, x.2.dim(3)], dtype: x.2.dtype)
                return (
                    concatenated([x.0, newData], axis: 2),
                    concatenated([x.1, newScales], axis: 2),
                    concatenated([x.2, newBiases], axis: 2)
                )
            }
            
            if var currentKeys = self.keys, var currentValues = self.values {
                // Trim to actual size if needed
                if prev % self.step != 0 {
                    currentKeys = (
                        currentKeys.0[.ellipsis, ..<prev, 0...],
                        currentKeys.1[.ellipsis, ..<prev, 0...],
                        currentKeys.2[.ellipsis, ..<prev, 0...]
                    )
                    currentValues = (
                        currentValues.0[.ellipsis, ..<prev, 0...],
                        currentValues.1[.ellipsis, ..<prev, 0...],
                        currentValues.2[.ellipsis, ..<prev, 0...]
                    )
                }
                
                self.keys = expandQuant(currentKeys)
                self.values = expandQuant(currentValues)
            } else {
                self.keys = initQuant(dim: kHeadDim)
                self.values = initQuant(dim: vHeadDim)
            }
        }
        
        self.offset += numSteps
        
        // Quantize the new keys and values
        let quantizedKeys = MLX.quantized(keys, groupSize: self.groupSize, bits: self.bits)
        let quantizedValues = MLX.quantized(values, groupSize: self.groupSize, bits: self.bits)
        
   
        // Store the quantized data
        for i in 0..<3 {
            let slice = prev ..< self.offset
            switch i {
            case 0:
                self.keys!.0[.ellipsis, slice, 0...] = quantizedKeys.0
                self.values!.0[.ellipsis, slice, 0...] = quantizedValues.0
            case 1:
                self.keys!.1[.ellipsis, slice, 0...] = quantizedKeys.1
                self.values!.1[.ellipsis, slice, 0...] = quantizedValues.1
            case 2:
                self.keys!.2[.ellipsis, slice, 0...] = quantizedKeys.2
                self.values!.2[.ellipsis, slice, 0...] = quantizedValues.2
            default:
                break
            }
        }
        
        // Return dummy arrays for protocol compatibility
        // The actual quantized data must be accessed via getQuantizedData()
        return (keys, values)
    }
    
    /// Get the stored quantized keys and values up to the current offset
    public func getQuantizedData() -> ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray))? {
        guard let keys = keys, let values = values else { return nil }
        
        // Return sliced tuples up to current offset
        let usedKeys = (
            keys.0[.ellipsis, ..<self.offset, 0...],
            keys.1[.ellipsis, ..<self.offset, 0...],
            keys.2[.ellipsis, ..<self.offset, 0...]
        )
        let usedValues = (
            values.0[.ellipsis, ..<self.offset, 0...],
            values.1[.ellipsis, ..<self.offset, 0...],
            values.2[.ellipsis, ..<self.offset, 0...]
        )
        
        return (usedKeys, usedValues)
    }
    
    public var debugDescription: String {
        let keysShape = keys.map { "(\($0.0.shape), \($0.1.shape), \($0.2.shape))" } ?? "-"
        let valuesShape = values.map { "(\($0.0.shape), \($0.1.shape), \($0.2.shape))" } ?? "-"
        return "QuantizedKVCache(offset: \(offset), step: \(step), bits: \(bits), groupSize: \(groupSize), keys: \(keysShape), values: \(valuesShape))"
    }
}