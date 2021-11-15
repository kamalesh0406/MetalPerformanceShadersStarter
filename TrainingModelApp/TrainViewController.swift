//
//  TrainViewController.swift
//  
//
//  Created by Kamalesh Palanisamy on 3/7/20.
//

import UIKit
import SwiftCSV
import CoreML
import Vision
import Charts

class TrainViewController: UIViewController {
    
    var model: MLModel?
    var currentTime: Int64?
    @IBOutlet weak var Chart: LineChartView!
    
    
    var imageConstraint: MLImageConstraint!
    var lineChartEntry =  [ChartDataEntry]()
    var imageLabelDict: [UIImage:String] = [:]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        //MARK: Load the CSV Data
        do{
            let resource: CSV? = try CSV(name: "test", extension: ".csv", bundle: .main , delimiter: ",", encoding: .utf8)
                print(resource!.namedRows)
            for row in resource!.namedRows{
                let image = UIImage(named: row["name"]!)!
                imageLabelDict[image] = row["class"]!
                print(imageLabelDict)
            }
        } catch {
            print("Some Error")
        }
        
        //MARK: Initialize the Models
        do{
            
            let fileManager = FileManager.default
            let documentDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            let fileURL  = documentDirectory.appendingPathComponent("NewoneUpdatable.mlmodelc")
            
            if let newModel = loadModel(url: fileURL){
                model = newModel
            }else{
                print("getting executed")
                if let modelURL = Bundle.main.url(forResource: "NewoneUpdatable", withExtension: "mlmodelc"){
                    if let newModel = loadModel(url: modelURL){
                        model = newModel
                    }
                }
            }
            if let model = model {
                imageConstraint = self.getImageConstraint(model: model)
            }
        }catch(let error){
            print("error is \(error.localizedDescription)")
        }
        
        
    }
    func getImageConstraint(model: MLModel) -> MLImageConstraint {
      return model.modelDescription.inputDescriptionsByName["image"]!.imageConstraint!
    }
    @IBAction func clickAction(_ sender: UIBarButtonItem) {
        //MARK: Start the training
        
        let modelconfig = MLModelConfiguration()
        modelconfig.computeUnits = .cpuAndGPU
        
        do{
            let fileManager = FileManager.default
            let documentDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            
            var modelURL = NewoneUpdatable.urlOfModelInThisBundle
            let pathOfFile = documentDirectory.appendingPathComponent("NewoneUpdatable.mlmodelc")
            
            if fileManager.fileExists(atPath: pathOfFile.path) {
                modelURL = pathOfFile
            }
            
            let progressHandler = {(context: MLUpdateContext) in
                print(context)
                switch context.event{
                case .trainingBegin:
                    self.currentTime = Date().toMillis()
                case .epochEnd:
                    
                    let epoch = context.metrics[.epochIndex]
                    let loss = context.metrics[.lossValue]
                    
                    let epochEndTime = Date().toMillis()
                    
                    print("Time for Training \(epochEndTime!-self.currentTime!)")
                    
                    self.currentTime = epochEndTime
                    let data = ChartDataEntry(x: epoch as! Double, y: loss as! Double)
                    self.lineChartEntry.append(data)
                default:
                    print("Unkown Event")
                }
            }
            let updateTask = try MLUpdateTask(forModelAt: modelURL, trainingData: batchProvider(), configuration: modelconfig , progressHandlers: MLUpdateProgressHandlers(forEvents: [.trainingBegin, .epochEnd], progressHandler: progressHandler){ (finalContext) in
                
                if finalContext.task.error?.localizedDescription == nil{
                    let fileManager = FileManager.default
                    do{
                        let documentDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                        let fileURL = documentDirectory.appendingPathComponent("NewoneUpdatable.mlmodelc")
                        try finalContext.model.write(to: fileURL)
                        
                        self.model = self.loadModel(url: fileURL)
                        
                        DispatchQueue.main.async{
                            let line1 = LineChartDataSet(entries: self.lineChartEntry, label: "Loss")
                            line1.colors = [NSUIColor.blue]
                            
                            let data = LineChartData()
                            data.addDataSet(line1)
                            data.setDrawValues(false)
                            
                            self.Chart.data = data
                            self.Chart.chartDescription?.text = "My awesome chart"
                        }
                    } catch(let error){
                        print("error is \(error.localizedDescription)")
                    }
                }
            })
            
            updateTask.resume()
            
        }catch(let error){
            print("error is \(error.localizedDescription)")
        }

    }

    //MARK: Load Model from URL
    private func loadModel(url: URL) -> MLModel?{
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            return try MLModel(contentsOf: url, configuration: config)
        } catch{
            print("Error loading model \(error)")
            return nil
        }
    }
    
    private func batchProvider() -> MLArrayBatchProvider{
        var batchInputs: [MLFeatureProvider] = []
        let imageOptions: [MLFeatureValue.ImageOption: Any] = [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue]
        
        for (image, className) in imageLabelDict{
            
            do{
                let featureValue = try MLFeatureValue(cgImage: image.cgImage!, constraint: imageConstraint, options: imageOptions)
                
                if let pixelBuffer = featureValue.imageBufferValue{
                    let x = NewoneUpdatableTrainingInput(image: pixelBuffer, classLabel: className)
                    batchInputs.append(x)
                }
            }catch (let error){
                print("error description is \(error.localizedDescription)")
            }
        }
        return MLArrayBatchProvider(array: batchInputs)
    }
    
}

extension Date {
    func toMillis() -> Int64! {
        return Int64(self.timeIntervalSince1970 * 1000)
    }
}
