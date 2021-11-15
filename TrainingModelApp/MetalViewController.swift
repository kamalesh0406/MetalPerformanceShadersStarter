//
//  MetalViewController.swift
//  TrainingModelApp
//
//  Created by Kamalesh Palanisamy on 3/27/20.
//  Copyright © 2020 Kamalesh Palanisamy. All rights reserved.
//

import UIKit
import GameplayKit
import MetalPerformanceShaders
import AVFoundation
import SocketIO
import Accelerate


typealias KernelSize = (width: Int, height: Int)

typealias Sample = (image:MPSImage, label:MPSCNNLossLabels)

typealias Batch = (image:[MPSImage], label:[MPSCNNLossLabels])

struct Gradients: Codable{
    let gradients: [Float]
}

let manager = SocketManager(socketURL: URL(string: "http://172.20.10.2:5000")!, config: [.log(false), .compress])
let socket = manager.defaultSocket

class DataLoader{
    
    public let channelFormat = MPSImageFeatureChannelFormat.float16
    public let imageHeight = 1
    public let imageWidth = 1
    public var featureChannels = 2
    public var imagePixelsCount = 2
    public let numberOfClasses = 1
    
    lazy var imageDescriptor: MPSImageDescriptor = {
        var imageDescriptor = MPSImageDescriptor(channelFormat: self.channelFormat, width: self.imageWidth, height: self.imageHeight, featureChannels: self.featureChannels)
        return imageDescriptor
    }()
    
    public var images = [Float]()
    public var labels = [Float]()
    
    public var count: Int{
        get{
            return self.images.count
        }
    }
    
    init?(images: [Float], label:Int, featureChannels: Int?){
        
        for image in images{
            self.images.append(image)
        }
        
        self.labels.append(Float(label))
        
        self.featureChannels = featureChannels!
        
        self.imagePixelsCount = featureChannels!
    }
    
    public func getImage(withDevice: MTLDevice, atIndex index:Int) -> MPSImage?{
        
        let image = MPSImage(device: withDevice, imageDescriptor: self.imageDescriptor)
        return setImageData(image, withDataFromIndex: index)
    }
    private func setImageData(_ image:MPSImage, withDataFromIndex index:Int) -> MPSImage?{
        let startIdx = index * self.imagePixelsCount
        let endIdx = startIdx + self.imagePixelsCount
        

        guard startIdx >= 0, endIdx <= self.images.count else{
            return nil
        }

        var imageData = [Float]()
        imageData += self.images[(startIdx..<endIdx)]
        
        let newData = float32toUint16(&imageData, count: imageData.count)
        newData.withUnsafeBufferPointer{ ptr in
            for i in 0..<image.texture.arrayLength{
                image.texture.replace(region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: 1, height: 1, depth: 1)), mipmapLevel: 0, slice: i, withBytes: ptr.baseAddress!.advanced(by: i*4), bytesPerRow: MemoryLayout<UInt16>.stride * 4, bytesPerImage: 0)
            }
        }
        return image
    }
    private func getLabel(withDevice device:MTLDevice,atIndex index:Int) -> MPSCNNLossLabels?{
        guard index>=0, index<self.images.count else{
            return nil
        }
        
        let labelValue = Float(self.labels[index])
        
        var labelVec = [Float](repeating: labelValue, count: self.numberOfClasses)
    
        let labelData = Data(fromArray: labelVec)
        
        guard let labelDesc = MPSCNNLossDataDescriptor(data: labelData, layout: MPSDataLayout.featureChannelsxHeightxWidth, size: MTLSize(width: 1, height: 1, depth: self.numberOfClasses)) else{
            return nil
        }
        
        let label = MPSCNNLossLabels(device: device, labelsDescriptor: labelDesc)
        label.label = "Label \(labelValue)"
        return label
    }
    public func getSample(withDevice device:MTLDevice, atIndex index:Int)  -> Sample?{
        guard let image = getImage(withDevice: device, atIndex: index), let label = getLabel(withDevice: device, atIndex: index) else{
            return nil
        }
        
        return Sample(image:image, label:label)
    }
    public func getSamples(withDevice device:MTLDevice) -> Batch?{
        var images = [MPSImage]()
        var labels = [MPSCNNLossLabels]()
        
        for i in 0..<self.labels.count{
            if Int(i) < 0 || Int(i) >= self.count{
                break
            }
            
            if let sample = self.getSample(withDevice: device, atIndex: i){
                images.append(sample.image)
                labels.append(sample.label)
            }
        }
        return Batch(image:images, label:labels)
    }
    
}


class MetalViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard let device = MTLCreateSystemDefaultDevice() else{
            fatalError("Metal Not Supported")
        }
        
        guard MPSSupportsMTLDevice(device) else{
            fatalError("MPS Not Supported")
        }
        
        guard let commandQueue = device.makeCommandQueue() else{
            fatalError("No command queue")
        }
        
        let inputs = [Float(1)]
        let sampleLabel = 0
        
        let inputdataLoader = DataLoader.init(images: inputs, label: sampleLabel, featureChannels: 1)
        let sample1 = inputdataLoader?.getSample(withDevice: device, atIndex: 0)
        
        
        let completeNetwork = CompleteNetworkRegression(withCommandQueue: commandQueue)
        let completeBenchmark = BenchMark()
        completeNetwork.train(withDataLoader: inputdataLoader!, epochs: 1){
            print("Complete Network Training done")
        }
        let end = completeBenchmark.timeTaken()
        print("Complete Network Time Taken", end, "ms")
        
        
        let firstNetwork = FirstNetwork(withCommandQueue: commandQueue)
        let benchmark = BenchMark()
        firstNetwork.predict(x: sample1!.image){(result) in
            print(result)
        }
 
        socket.on("receive_inputs"){data, ack in
            
            print(data)
            
            var inputs: [Float] = []
            var label: Int = 0
            
            if let values = data as? Array<Any>{
                for element in values{
                    let newData = element as! Dictionary<String, Any>
                    
                    inputs = newData["data"] as! Array<Float>
                    label = newData["label"] as! Int
                }
            }
            
            guard let dataLoader = DataLoader(images:inputs as [Float] , label: label as Int, featureChannels: 3) else{
                 fatalError("DataLoader is not created")
             }

            let finalNetwork = FinalNetwork(withCommandQueue: commandQueue)

            finalNetwork.train(withDataLoader: dataLoader){
                print("Training Done")
            }
            
            finalNetwork.predict(x: dataLoader.getSample(withDevice: device, atIndex: 0)!.image){ (result) in
                print(result, "New Result")
            }
            
            
        }
        
        socket.on("receive_gradients"){data, ack in
            print(data)
            
            var gradients: [Float] = []
            if let values = data as? Array<Any>{
                for element in values{
                    let newData = element as! Dictionary<String, Any>
                    do {
                      gradients = newData["data"] as! Array<Float>
                    }catch{
                        print("Error")
                        socket.emit("gradients_error", ["data":1])
                    }
                }
            }
            
            let inputs = sample1!.image.toFloatArray()
            
            let multiplication = MatrixMultiplication(device: device, matrixA: gradients, matrixB: inputs!, rowsA: 3, columnsA: 1, rowsB: 1, columnsB: 1)
            let result = multiplication.multiply()
            var newWeights: [Float] = []
            
            let oldWeights = firstNetwork.dataSources[0].weightsAndBiasesState?.weights.toArray(type: Float.self)
            for i in 0..<oldWeights!.count{
                newWeights.append(oldWeights![i] + result[i])
            }
            
            let fcDesc = firstNetwork.dataSources[0]
            let weightsBuffer = device.makeBuffer(bytes: newWeights, length: fcDesc.inputFeatureChannels * fcDesc.outputFeatureChannels * MemoryLayout<Float>.stride, options: [])
            let biasBuffer = device.makeBuffer(bytes: fcDesc.biasTermsData!.toArray(type: Float.self), length: fcDesc.outputFeatureChannels*MemoryLayout<Float>.stride, options: [])
            fcDesc.weightsAndBiasesState = MPSCNNConvolutionWeightsAndBiasesState(weights: weightsBuffer!, weightsOffset: 0, biases: biasBuffer, biasesOffset: 0, cnnConvolutionDescriptor: fcDesc.descriptor())
            
            firstNetwork.dataSources[0] = fcDesc
            let end = benchmark.timeTaken()
            print("The time taken for the 1 epoch is", end, "ms")
            
            firstNetwork.inferenceGraph = firstNetwork.createInferenceGraph(mode: "Second")
//            firstNetwork.predict(x: sample1!.image){(result) in
//                print(result, "First Network")
//            }
            
        }
        
        
        socket.connect()
        
        // Do any additional setup after loading the view.
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
