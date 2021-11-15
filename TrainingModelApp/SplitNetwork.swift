//
//  Network.swift
//  TrainingModelApp
//
//  Created by Kamalesh Palanisamy on 4/2/20.
//  Copyright © 2020 Kamalesh Palanisamy. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class FirstNetwork{
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    
    var trainingGraph: MPSNNGraph?
    var inferenceGraph: MPSNNGraph?
    
    var dataSources = [DataSource]()
    
    init(withCommandQueue commandQueue: MTLCommandQueue){
        self.device = commandQueue.device
        self.commandQueue = commandQueue
        
        self.inferenceGraph = self.createInferenceGraph()
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
                    self.sendValuestoServer(inputs: outputImage!.toFloatArray()!)
                    handler(outputImage!.toFloatArray())
                    return
                }
                
                handler(nil)
            }
            
        }
    }
    
    private func sendValuestoServer(inputs: [Float]){
        socket.emit("output_from_first", ["data":inputs])
        
        DispatchQueue.global(qos: .background).async {
            socket.emit("outputs_from_first", ["data":inputs])
        }
        
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
    
    public func createInferenceGraph(mode: String = "initial") -> MPSNNGraph?{
        
        let input = MPSNNImageNode(handle: nil)
        
        var fcDesc: DataSource
        
        if (mode == "initial"){
          fcDesc = DataSource(name: "fc_1", kernelSize: KernelSize(width: 1, height: 1), inputFeatureChannels: 1, outputFeatureChannels: 3)
          let weightsBuffer = device.makeBuffer(bytes: (fcDesc.weightsData!.toArray(type: Float.self)), length: fcDesc.inputFeatureChannels * fcDesc.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
          let biasBuffer = device.makeBuffer(bytes: fcDesc.biasTermsData!.toArray(type: Float.self), length: fcDesc.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
          fcDesc.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer!, weightsOffset: 0, biases: biasBuffer!, biasesOffset: 0, cnnConvolutionDescriptor: fcDesc.cnnDescriptor)
          
          self.dataSources.append(fcDesc)
        }else{
            fcDesc = self.dataSources[0]
        }

        let fc = MPSCNNFullyConnectedNode(source: input, weights: fcDesc)
        let sigmoid = MPSCNNNeuronSigmoidNode(source: fc.resultImage)
        
        let graph = MPSNNGraph(device: self.device, resultImage: sigmoid.resultImage, resultImageIsNeeded: true)
        
        return graph
    }
    
}

class FinalNetwork{
    enum NetworkMode{
        case training
        case inference
    }
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    var mode: NetworkMode = NetworkMode.training
    
    var trainingGraph: MPSNNGraph?
    var inferenceGraph: MPSNNGraph?
    
    var dataSources = [DataSource]()
    
    init(withCommandQueue commandQueue: MTLCommandQueue, mode:NetworkMode=NetworkMode.training){
        self.device = commandQueue.device
        self.commandQueue = commandQueue
        self.mode = mode
        
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
                sampleImage += samples.image[0].toFloatArray()![0..<3]
            }
            
            print("Epoch \(e)")
            let initialweights = initialWeights[0]
            initialWeights[0] = self.dataSources[0].updateAndSendParametersToServer(device: device, initialWeights: initialweights, inputs: sampleImage)!
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

        let fcDatasource0 = DataSource(name: "fc_0", kernelSize: KernelSize(width: 1, height: 1), inputFeatureChannels: 3, outputFeatureChannels: 3, optimizer: optimizer)
        let weightsBuffer0 = device.makeBuffer(bytes: (fcDatasource0.weightsData!.toArray(type: Float.self)), length: fcDatasource0.inputFeatureChannels * fcDatasource0.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
         let biasBuffer0 = device.makeBuffer(bytes: fcDatasource0.biasTermsData!.toArray(type: Float.self), length: fcDatasource0.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
         fcDatasource0.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer0!, weightsOffset: 0, biases: biasBuffer0!, biasesOffset: 0, cnnConvolutionDescriptor: fcDatasource0.cnnDescriptor)
        self.dataSources.append(fcDatasource0)
        
        let fc0 = MPSCNNFullyConnectedNode(source: input, weights: fcDatasource0)
        let beforefirstSigmoid = MPSCNNNeuronSigmoidNode(source: fc0.resultImage)
        
        let fcDatasource = DataSource(name: "fc_1", kernelSize: KernelSize(width: 1, height: 1), inputFeatureChannels: 3, outputFeatureChannels: 3, optimizer: optimizer)
        let weightsBuffer = device.makeBuffer(bytes: (fcDatasource.weightsData!.toArray(type: Float.self)), length: fcDatasource.inputFeatureChannels * fcDatasource.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
         let biasBuffer = device.makeBuffer(bytes: fcDatasource.biasTermsData!.toArray(type: Float.self), length: fcDatasource.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
         fcDatasource.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer!, weightsOffset: 0, biases: biasBuffer!, biasesOffset: 0, cnnConvolutionDescriptor: fcDatasource.cnnDescriptor)
        self.dataSources.append(fcDatasource)
        
        let fc = MPSCNNFullyConnectedNode(source: beforefirstSigmoid.resultImage, weights: fcDatasource)
        let firstSigmoid = MPSCNNNeuronSigmoidNode(source: fc.resultImage)
        
        let fcDatasource2 = DataSource(name: "fc_2", kernelSize: KernelSize(width: 1, height: 1), inputFeatureChannels: 3, outputFeatureChannels: 1, optimizer: optimizer)
        let weightsBuffer2 = device.makeBuffer(bytes: (fcDatasource2.weightsData!.toArray(type: Float.self)), length: fcDatasource2.inputFeatureChannels * fcDatasource2.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
          let biasBuffer2 = device.makeBuffer(bytes: fcDatasource2.biasTermsData!.toArray(type: Float.self), length: fcDatasource2.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
          fcDatasource2.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer2!, weightsOffset: 0, biases: biasBuffer2!, biasesOffset: 0, cnnConvolutionDescriptor: fcDatasource2.cnnDescriptor)

        self.dataSources.append(fcDatasource2)
        
        let fc2 = MPSCNNFullyConnectedNode(source: firstSigmoid.resultImage, weights: fcDatasource2)
        
        let finalSigmoid = MPSCNNNeuronSigmoidNode(source: fc2.resultImage)
        
        let lossDesc = MPSCNNLossDescriptor(type: MPSCNNLossType.meanSquaredError, reductionType: MPSCNNReductionType.mean)
        
        let loss = MPSCNNLossNode(source: finalSigmoid.resultImage, lossDescriptor: lossDesc)
        
        
        let sigmoidG = finalSigmoid.gradientFilter(withSource: loss.resultImage)
        
        let fc2G = fc2.gradientFilter(withSource: sigmoidG.resultImage) as! MPSCNNConvolutionGradientNode
        fc2G.trainingStyle = MPSNNTrainingStyle.updateDeviceGPU
        
        let firstsigmoidG = firstSigmoid.gradientFilter(withSource: fc2G.resultImage)
        
        let fcG = fc.gradientFilter(withSource: firstsigmoidG.resultImage) as! MPSCNNConvolutionGradientNode
        fcG.trainingStyle = MPSNNTrainingStyle.updateDeviceGPU
        
        let beforefirstSigmoidG = beforefirstSigmoid.gradientFilter(withSource: fcG.resultImage)
        
        let fc0G = fc0.gradientFilter(withSource: beforefirstSigmoidG.resultImage) as! MPSCNNConvolutionGradientNode
        fc0G.trainingStyle = MPSNNTrainingStyle.updateDeviceGPU
        
        let graph = MPSNNGraph(device: self.device, resultImage: fc0G.resultImage, resultImageIsNeeded: false)
        
        return graph
        
    }
    
    private func createInferenceGraph() -> MPSNNGraph?{
        
        let input = MPSNNImageNode(handle: nil)
        
        let fc0Desc = self.dataSources[0]

        let fc0 = MPSCNNFullyConnectedNode(source: input, weights: fc0Desc)
        let sigmoid0 = MPSCNNNeuronSigmoidNode(source: fc0.resultImage)
        
        let fcDesc = self.dataSources[1]

        let fc = MPSCNNFullyConnectedNode(source: sigmoid0.resultImage, weights: fcDesc)
        let sigmoid = MPSCNNNeuronSigmoidNode(source: fc.resultImage)
        
        let fcDesc2 = self.dataSources[2]
        
        let fc2 = MPSCNNFullyConnectedNode(source: sigmoid.resultImage, weights: fcDesc2)
        let finalSigmoid = MPSCNNNeuronSigmoidNode(source: fc2.resultImage)
        
        let graph = MPSNNGraph(device: self.device, resultImage: finalSigmoid.resultImage, resultImageIsNeeded: true)
        
        return graph
    }
    
}
