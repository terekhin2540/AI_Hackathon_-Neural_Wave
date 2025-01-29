//
//  MonitoringView.swift
//  NeuraWave
//
//  Created by Roman Rakhlin on 10/27/24.
//

import SwiftUI

struct MonitoringView: View {
    
    private let samples: [Sample] = [
        Sample(id: 0, preview: Image("previewOne"), videoName: "videoOne"),
        Sample(id: 1, preview: Image("previewTwo"), videoName: "videoTwo"),
        Sample(id: 2, preview: Image("previewThree"), videoName: "videoThree")
    ]
    
    @State private var selectedSample: Sample? = nil
    
    var body: some View {
        ZStack {
            Color("background").edgesIgnoringSafeArea(.all)
            
            VStack(spacing: 28) {
                HStack {
                    Text("Monitoring")
                        .font(.system(size: 32, weight: .black))
                    
                    Spacer()
                    
                    Text("❤️ Made with Love by RadYOmki")
                        .font(.system(size: 14, weight: .light))
                }
                
                ZStack {
                    GeometryReader { geometry in
                        HStack {
                            ForEach(samples, id: \.id) { sample in
                                VStack(spacing: 0) {
                                    ZStack {
                                        ZStack {
                                            sample.preview
                                                .resizable()
                                                .scaledToFill()
                                            
                                            Color.black.opacity(0.2)
                                        }
                                        .frame(width: geometry.size.width / 3 - 20, height: geometry.size.height * 0.5)
                                        .clipped()
                                        
                                        Image(systemName: "play.circle")
                                            .font(.system(size: 36, weight: .semibold))
                                            .foregroundColor(.white)
                                    }
                                    
                                    HStack {
                                        Spacer()
                                        
                                        Text("SAMPLE \(sample.id + 1)")
                                            .font(.system(size: 12, weight: .semibold))
                                            .opacity(0.9)
                                        
                                        Spacer()
                                    }
                                    .padding(.top, 6)
                                    .padding(.bottom, 4)
                                    .padding(.horizontal, 6)
                                    .background(Color.white.opacity(0.05))
                                }
                                .fixedSize(horizontal: true, vertical: false)
                                .cornerRadius(8)
                                .shadow(radius: 16)
                                .onTapGesture {
                                    withAnimation {
                                        selectedSample = sample
                                    }
                                }
                                
                                if sample.id != 2 {
                                    Spacer()
                                }
                            }
                        }
                    }
                    
                    if selectedSample != nil {
                        SampleView(sample: $selectedSample)
                            .cornerRadius(6)
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 36)
            .padding(.bottom, 20)
        }
    }
}

#Preview {
    MonitoringView()
}
