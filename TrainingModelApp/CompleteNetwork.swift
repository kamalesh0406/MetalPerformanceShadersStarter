//
//  CompleteNetwork.swift
//  TrainingModelApp
//
//  Created by Kamalesh Palanisamy on 4/11/20.
//  Copyright © 2020 Kamalesh Palanisamy. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class CompleteNetworkRegression{
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    var dataSources = [DataSource]()
    var trainingGraph: MPSNNGraph?
    var inferenceGraph: MPSNNGraph?
    
    init(withCommandQueue commandQueue: MTLCommandQueue) {
        self.device = commandQueue.device
        self.commandQueue = commandQueue
        self.trainingGraph = self.createTrainingGraph()
        self.inferenceGraph = self.createInferenceGraph()
    }
    
    public func train(withDataLoader dataLoader: DataLoader, epochs: Int = 1, completionHandler handler: @escaping () -> Void) -> Bool{
        
        var initialWeights : [[Float]] = []
        

        initialWeights.append(self.dataSources[0].weightsAndBiasesState!.weights.toArray(type: Float.self))

        for e in 0..<epochs{
            var sampleImage: [Float] = []
            if let samples = dataLoader.getSamples(withDevice: device){
                self.trainStep(images: samples.image, labels: samples.label)
                sampleImage += samples.image[0].toFloatArray()!
            }
            
            print("Epoch \(e)")
        }
        
        
        handler()
        
        return true
    }
    
    public func predict(x: MPSImage, completionHandler handler: @escaping ([Float]?) -> Void) -> Void{
        
        guard let graph = self.inferenceGraph else{
            return
        }
        
        
        graph.executeAsync(withSourceImages: [x]){ (outputImage, error) in
            
            DispatchQueue.main.async {
                if error != nil {
                    print(error!)
                    handler(nil)
                    return
                }
                
                if outputImage != nil {
                    handler(outputImage!.toFloatArray())
                    return
                }
                
                handler(nil)
            }
            
        }
    }
    
    private func trainStep(images: [MPSImage], labels: [MPSCNNLossLabels]){
        
        guard let graph = self.trainingGraph else{
            return
        }
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
            return
        }
        
        graph.encodeBatch(to: commandBuffer, sourceImages: [images], sourceStates: [labels])
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        
    }
    
    private func updateDataSources(){
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else{
            return
        }
        
        for datasource in self.dataSources{
            datasource.synchronizeParameters(on: commandBuffer)
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
    }
    private func createTrainingGraph() -> MPSNNGraph?{
        let input = MPSNNImageNode(handle: nil)
        
        let optimizer = MPSNNOptimizerStochasticGradientDescent(device: self.device, learningRate: 0.1)

        let fcDatasource0 = DataSource(name: "fc_0", kernelSize: KernelSize(width: 1, height: 1), inputFeatureChannels: 1, outputFeatureChannels: 3, optimizer: optimizer)
        let weightsBuffer0 = device.makeBuffer(bytes: (fcDatasource0.weightsData!.toArray(type: Float.self)), length: fcDatasource0.inputFeatureChannels * fcDatasource0.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
         let biasBuffer0 = device.makeBuffer(bytes: fcDatasource0.biasTermsData!.toArray(type: Float.self), length: fcDatasource0.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
         fcDatasource0.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer0!, weightsOffset: 0, biases: biasBuffer0!, biasesOffset: 0, cnnConvolutionDescriptor: fcDatasource0.cnnDescriptor)
        self.dataSources.append(fcDatasource0)
        
        let fc0 = MPSCNNFullyConnectedNode(source: input, weights: fcDatasource0)
        let sigmoid0 = MPSCNNNeuronSigmoidNode(source: fc0.resultImage)
        
        let fcDatasource1 = DataSource(name: "fc_1", kernelSize: KernelSize(width: 1, height: 1), inputFeatureChannels: 3, outputFeatureChannels: 3, optimizer: optimizer)
        let weightsBuffer1 = device.makeBuffer(bytes: (fcDatasource1.weightsData!.toArray(type: Float.self)), length: fcDatasource1.inputFeatureChannels * fcDatasource1.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
         let biasBuffer1 = device.makeBuffer(bytes: fcDatasource1.biasTermsData!.toArray(type: Float.self), length: fcDatasource1.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
         fcDatasource1.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer1!, weightsOffset: 0, biases: biasBuffer1!, biasesOffset: 0, cnnConvolutionDescriptor: fcDatasource1.cnnDescriptor)
        self.dataSources.append(fcDatasource1)
        
        let fc1 = MPSCNNFullyConnectedNode(source: sigmoid0.resultImage, weights: fcDatasource1)
        let sigmoid1 = MPSCNNNeuronSigmoidNode(source: fc1.resultImage)
        
        let fcDatasource2 = DataSource(name: "fc_2", kernelSize: KernelSize(width: 1, height: 1), inputFeatureChannels: 3, outputFeatureChannels: 3, optimizer: optimizer)
        let weightsBuffer2 = device.makeBuffer(bytes: (fcDatasource2.weightsData!.toArray(type: Float.self)), length: fcDatasource2.inputFeatureChannels * fcDatasource2.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
         let biasBuffer2 = device.makeBuffer(bytes: fcDatasource2.biasTermsData!.toArray(type: Float.self), length: fcDatasource2.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
         fcDatasource2.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer2!, weightsOffset: 0, biases: biasBuffer2!, biasesOffset: 0, cnnConvolutionDescriptor: fcDatasource2.cnnDescriptor)
        self.dataSources.append(fcDatasource2)
        
        let fc2 = MPSCNNFullyConnectedNode(source: sigmoid1.resultImage, weights: fcDatasource2)
        let sigmoid2 = MPSCNNNeuronSigmoidNode(source: fc2.resultImage)
        
        let fcDatasource3 = DataSource(name: "fc_3", kernelSize: KernelSize(width: 1, height: 1), inputFeatureChannels: 3, outputFeatureChannels: 1, optimizer: optimizer)
        let weightsBuffer3 = device.makeBuffer(bytes: (fcDatasource3.weightsData!.toArray(type: Float.self)), length: fcDatasource3.inputFeatureChannels * fcDatasource3.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
         let biasBuffer3 = device.makeBuffer(bytes: fcDatasource3.biasTermsData!.toArray(type: Float.self), length: fcDatasource3.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
         fcDatasource3.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer3!, weightsOffset: 0, biases: biasBuffer3!, biasesOffset: 0, cnnConvolutionDescriptor: fcDatasource3.cnnDescriptor)
        self.dataSources.append(fcDatasource3)
        
        let fc3 = MPSCNNFullyConnectedNode(source: sigmoid2.resultImage, weights: fcDatasource3)
        let sigmoid3 = MPSCNNNeuronSigmoidNode(source: fc3.resultImage)
        
        let lossDesc = MPSCNNLossDescriptor(type: .meanSquaredError, reductionType: .mean)
        let lossNode = MPSCNNLossNode(source: sigmoid3.resultImage, lossDescriptor: lossDesc)
        
        let sigmoid3G = sigmoid3.gradientFilter(withSource: lossNode.resultImage)
        let fc3G = fc3.gradientFilter(withSource: sigmoid3G.resultImage) as! MPSCNNConvolutionGradientNode
        fc3G.trainingStyle = .updateDeviceGPU
        
        let sigmoid2G = sigmoid2.gradientFilter(withSource: fc3G.resultImage)
        let fc2G = fc2.gradientFilter(withSource: sigmoid2G.resultImage) as! MPSCNNConvolutionGradientNode
        fc2G.trainingStyle = .updateDeviceGPU
        
        let sigmoid1G = sigmoid1.gradientFilter(withSource: fc2G.resultImage)
        let fc1G = fc1.gradientFilter(withSource: sigmoid1G.resultImage) as! MPSCNNConvolutionGradientNode
        fc1G.trainingStyle = .updateDeviceGPU
        
        let sigmoid0G = sigmoid0.gradientFilter(withSource: fc2G.resultImage)
        let fc0G = fc0.gradientFilter(withSource: sigmoid0G.resultImage) as! MPSCNNConvolutionGradientNode
        fc0G.trainingStyle = .updateDeviceGPU
        
        let graph = MPSNNGraph(device: device, resultImage: fc0G.resultImage, resultImageIsNeeded: false)
        
        return graph
        
    }
    
    public func createInferenceGraph() -> MPSNNGraph?{
        let input = MPSNNImageNode(handle: nil)
        
        let fc0 = MPSCNNFullyConnectedNode(source: input, weights: self.dataSources[0])
        let sigmoid0 = MPSCNNNeuronSigmoidNode(source: fc0.resultImage)
        
        let fc1 = MPSCNNFullyConnectedNode(source: sigmoid0.resultImage, weights: self.dataSources[1])
        let sigmoid1 = MPSCNNNeuronSigmoidNode(source: fc1.resultImage)
        
        let fc2 = MPSCNNFullyConnectedNode(source: sigmoid1.resultImage, weights: self.dataSources[2])
        let sigmoid2 = MPSCNNNeuronSigmoidNode(source: fc2.resultImage)
        
        let fc3 = MPSCNNFullyConnectedNode(source: sigmoid2.resultImage, weights: self.dataSources[3])
        let sigmoid3 = MPSCNNNeuronSigmoidNode(source: fc3.resultImage)
        
        let graph = MPSNNGraph(device: device, resultImage: sigmoid3.resultImage, resultImageIsNeeded: true)
        
        return graph
    }
}
