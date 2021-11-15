//
//  MatrixMultiplication.swift
//  TrainingModelApp
//
//  Created by Kamalesh Palanisamy on 4/4/20.
//  Copyright © 2020 Kamalesh Palanisamy. All rights reserved.
//

import Foundation
import MetalPerformanceShaders


class MatrixMultiplication{
    
    var matrixAentries: [Float]
    var matrixBentries: [Float]
    
    var device: MTLDevice
    
    var rowsA: Int
    var columnsA: Int
    
    var rowsB: Int
    var columnsB: Int
    
    init(device: MTLDevice, matrixA: [Float], matrixB: [Float], rowsA: Int, columnsA: Int, rowsB: Int, columnsB: Int) {
        self.device = device
        
        self.matrixAentries = matrixA
        self.matrixBentries = matrixB
        
        self.rowsA = rowsA
        self.columnsA = columnsA
        
        self.rowsB = rowsB
        self.columnsB = columnsB
        
    }
    
    func multiply() -> [Float]{
            let commandQueue = device.makeCommandQueue()!
            let commandBuffer = commandQueue.makeCommandBuffer()!

        let mmKernel = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: rowsA, resultColumns: columnsB, interiorColumns: columnsA, alpha: 1.0, beta: 0.0)

            let totalBytesA = MemoryLayout<Float>.stride * matrixAentries.count
            let bufferA = device.makeBuffer(bytes: matrixAentries, length: totalBytesA, options: .storageModeShared)
        let descriptorA = MPSMatrixDescriptor(rows: rowsA, columns: columnsA, rowBytes: totalBytesA/rowsA, dataType: .float32)

            let A = MPSMatrix(buffer: bufferA!, descriptor: descriptorA)

            let totalBytesB = MemoryLayout<Float>.stride * matrixBentries.count
            let bufferB = device.makeBuffer(bytes: matrixBentries, length: totalBytesB, options: .storageModeShared)
            let descriptorB = MPSMatrixDescriptor(rows: rowsB, columns: columnsB, rowBytes: totalBytesB/rowsB, dataType: .float32)


            let B = MPSMatrix(buffer: bufferB!, descriptor: descriptorB)

            let totalBytesC = MemoryLayout<Float>.stride * A.rows * B.columns
            let bufferC = device.makeBuffer(length: totalBytesC, options: .storageModeShared)
            let descriptorC = MPSMatrixDescriptor(rows: A.rows, columns: B.columns, rowBytes: totalBytesC/A.rows, dataType: .float32)

            let C = MPSMatrix(buffer: bufferC!, descriptor: descriptorC)

            mmKernel.encode(commandBuffer: commandBuffer, leftMatrix: A, rightMatrix: B, resultMatrix: C)
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            var output = [Float]()

            let rawPointer = C.data.contents()
            let typePointer = rawPointer.bindMemory(to: Float.self, capacity: A.rows * B.columns)
            let bufferPointer = UnsafeBufferPointer(start: typePointer, count: A.rows*B.columns)

            bufferPointer.map { value in
                output += [value]
            }

            return output

    }
}
