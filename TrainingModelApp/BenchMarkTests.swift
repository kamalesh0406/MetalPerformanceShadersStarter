//
//  BenchMarkTests.swift
//  TrainingModelApp
//
//  Created by Kamalesh Palanisamy on 4/11/20.
//  Copyright © 2020 Kamalesh Palanisamy. All rights reserved.
//

import Foundation

class BenchMark{
    var startTime: Int
    var endTime: Int
    
    init() {
        self.startTime = Int(Date().timeIntervalSince1970 * 1000)
        self.endTime = 0
    }
    
    public func timeTaken() -> Int{
        self.endTime = Int(NSDate().timeIntervalSince1970 * 1000)
        return(self.endTime - self.startTime)
    }
}
