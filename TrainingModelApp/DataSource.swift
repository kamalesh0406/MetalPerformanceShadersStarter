import Foundation
import MetalPerformanceShaders
import GameplayKit

class DataSource: NSObject, MPSCNNConvolutionDataSource{
    
    let name: String
    let kernelSize: KernelSize
    let inputFeatureChannels: Int
    let outputFeatureChannels: Int
    
    var optimizer: MPSNNOptimizerStochasticGradientDescent?
    var weightsAndBiasesState: MPSCNNConvolutionWeightsAndBiasesState?
    
    var weightsData: Data!
    var biasTermsData: Data!
    var weightsServer: [Float]?
    
    var mode: String
    
    lazy var cnnDescriptor : MPSCNNConvolutionDescriptor = {
        let descriptor = MPSCNNConvolutionDescriptor(kernelWidth: self.kernelSize.width, kernelHeight: self.kernelSize.height, inputFeatureChannels: self.inputFeatureChannels, outputFeatureChannels: self.outputFeatureChannels)
        return descriptor
    }()
    init(name: String, kernelSize: KernelSize, inputFeatureChannels: Int, outputFeatureChannels: Int, optimizer:MPSNNOptimizerStochasticGradientDescent? = nil, mode:String = "normal", weightsServer:[Float]? = nil) {
        self.name = name
        self.kernelSize = kernelSize
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.optimizer = optimizer
        self.mode = mode
        if (weightsServer != nil) {
            self.weightsServer = weightsServer
        }
        
        super.init()
        self.weightsData = generateRandomWeights()
        self.biasTermsData  = generateBiasData()
        
    }
    
    func dataType() -> MPSDataType {
        return MPSDataType.float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return self.cnnDescriptor
    }
    
    public func weights() -> UnsafeMutableRawPointer {
        return UnsafeMutableRawPointer(mutating: (self.weightsData! as NSData).bytes)
    }
    
    public func biasTerms() -> UnsafeMutablePointer<Float>? {
        guard let biasTermsData = self.biasTermsData else {
            return nil
        }
        
        return UnsafeMutableRawPointer(mutating: (biasTermsData as NSData).bytes).bindMemory(to: Float.self, capacity: self.outputFeatureChannels * MemoryLayout<Float>.size)
    }
    
    func load() -> Bool {
        
        self.weightsData = generateRandomWeights()
        
        return self.weightsData != nil
    }
    
    func purge() {
        
        self.weightsData = generateRandomWeights()
        self.biasTermsData = generateBiasData()
    }
    
    func label() -> String? {
        return self.name
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        let copy = DataSource(name: self.name, kernelSize: self.kernelSize, inputFeatureChannels: self.inputFeatureChannels, outputFeatureChannels: self.outputFeatureChannels, optimizer: self.optimizer)
        
        copy.weightsAndBiasesState = self.weightsAndBiasesState
        return copy as Any
    }
    
    private func generateRandomWeights() -> Data?{
        let count = self.kernelSize.width * self.kernelSize.height * self.inputFeatureChannels * self.outputFeatureChannels
        
        var weights = Array<Float>(repeating: 0, count: count)
        let random = GKRandomSource()
        
        for i in 0..<self.outputFeatureChannels{
            for j in 0..<self.inputFeatureChannels{
                let index = i * self.inputFeatureChannels + j
                weights[index] = random.nextUniform()
            }
        }
        
        return Data(fromArray: weights)
    }
    private func generateBiasData() -> Data?{
        var bias = Array<Float>(repeating: 0, count: self.outputFeatureChannels)
        
        return Data(fromArray: bias)
    }
}

extension DataSource{
    
    func update(with gradientState: MPSCNNConvolutionGradientState, sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> Bool {
        return false
    }
    func update(with commandBuffer: MTLCommandBuffer, gradientState: MPSCNNConvolutionGradientState, sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        guard let optimizer = self.optimizer, let weightsAndBiasesState = self.weightsAndBiasesState else{
            return nil
        }
        
        optimizer.encode(commandBuffer: commandBuffer, convolutionGradientState: gradientState, convolutionSourceState: sourceState, inputMomentumVectors: nil, resultState: weightsAndBiasesState)
        
        return self.weightsAndBiasesState
    }
    func synchronizeParameters(on commandBuffer: MTLCommandBuffer){
        guard let weightsAndBiasesState = self.weightsAndBiasesState else{
            return
        }
        
        weightsAndBiasesState.synchronize(on: commandBuffer)
    }
    func updateAndSendParametersToServer(device:MTLDevice? , initialWeights: [Float]? = nil, inputs: [Float]? = nil) -> [Float]?{
        print("Trying to send the parameters to the amazing server")
        
        
        guard let weightsAndBiasesState = self.weightsAndBiasesState else{
            return nil
        }
        
        self.weightsData = Data(fromArray: weightsAndBiasesState.weights.toArray(type: Float.self))
        
        if let biasData = weightsAndBiasesState.biases{
            self.biasTermsData = Data(fromArray: biasData.toArray(type: Float.self))
        }
        
        var diffgradients: [Float] = []
        var newWeights: [Float] = []

        if initialWeights != nil{
            newWeights = weightsAndBiasesState.weights.toArray(type: Float.self)
            for i in 0..<newWeights.count{
                diffgradients.append(newWeights[i] - initialWeights![i])
            }
            
        }
        if (self.outputFeatureChannels != 1){
          let multiplication = MatrixMultiplication(device: device!, matrixA: initialWeights!, matrixB: diffgradients, rowsA: 3, columnsA: 3, rowsB: 3, columnsB: 3)
          
          let result = multiplication.multiply()
        
            var differentiationInput = Array<Float>(repeating: 1, count: inputs!.count)
            
            for i in 0..<differentiationInput.count{
                differentiationInput[i] = differentiationInput[i] - inputs![i]
            }
            
            let multiplication2 = MatrixMultiplication(device: device!, matrixA: result, matrixB: differentiationInput, rowsA: 3, columnsA: 3, rowsB: 3, columnsB: 1)
            var resultfinal = multiplication2.multiply()
            
            DispatchQueue.global(qos: .background).async {
                socket.emit("gradients_from_previous", ["data":resultfinal])
                print("Emitted")
            }
            
        }
        
        return newWeights
    
    }
}//
//  DataSource.swift
//  TrainingModelApp
//
//  Created by Kamalesh Palanisamy on 3/31/20.
//  Copyright © 2020 Kamalesh Palanisamy. All rights reserved.
//

