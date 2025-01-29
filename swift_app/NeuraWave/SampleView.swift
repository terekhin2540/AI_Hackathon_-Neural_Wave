//
//  SampleView.swift
//  NeuraWave
//
//  Created by Roman Rakhlin on 10/27/24.
//

import SwiftUI

struct SampleView: View {
    
    @Binding var sample: Sample?
    
    var body: some View {
        ZStack {
            Color("background").edgesIgnoringSafeArea(.all)
            
            VStack {
                HStack {
                    Text("Close")
                        .font(.system(size: 16, weight: .semibold))
                        .onTapGesture {
                            withAnimation {
                                sample = nil
                            }
                        }
                    
                    Spacer()
                }
                
                Spacer()
                
                if let sample {
                    ExtVideoPlayer(fileName: sample.videoName)
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
        }
    }
}

//#Preview {
//    SampleView()
//}
