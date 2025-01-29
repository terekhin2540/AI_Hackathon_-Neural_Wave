//
//  PlayerView.swift
//
//
//  Created by Igor Shelopaev on 10.02.2023.
//

import SwiftUI
import Combine
import AVKit

/// Player view for running a video in loop
@available(iOS 14.0, macOS 11.0, tvOS 14.0, *)
public struct ExtVideoPlayer: View {
    
    @State private var processedImage: NSImage?
    @State private var isAligned = false
    /// Set of settings for video the player
    @Binding public var settings: VideoSettings
    
    /// Binding to a playback command that controls playback actions
    @Binding public var command: PlaybackCommand
    
    /// The current playback time, represented as a Double.
    @State private var currentTime: Double = 0.0
    
    /// The current state of the player event,
    @State private var playerEvent: [PlayerEvent] = []

    /// A publisher that emits the current playback time as a `Double`. It is initialized privately within the view.
    @State private var timePublisher = PassthroughSubject<Double, Never>()
    
    @State private var snapshotPublisher = PassthroughSubject<NSImage, Never>()

    /// A publisher that emits player events as `PlayerEvent` values. It is initialized privately within the view.
    @State private var eventPublisher = PassthroughSubject<PlayerEvent, Never>()
    
    @State private var nsImage: NSImage?
    
    // MARK: - Life cycle
    
    /// Player initializer
    /// - Parameters:
    ///   - fileName: The name of the video file.
    ///   - ext: The file extension, with a default value of "mp4".
    ///   - gravity: The video gravity setting, with a default value of `.resizeAspect`.
    ///   - timePublishing: An optional `CMTime` value for time publishing, with a default value of 1 second.
    ///   - eColor: The color to be used, with a default value of `.accentColor`.
    ///   - eFontSize: The font size to be used, with a default value of 17.0.
    ///   - command: A binding to the playback command, with a default value of `.play`.
    public init(
        fileName: String,
        ext: String = "mp4",
        gravity: AVLayerVideoGravity = .resizeAspect,
        timePublishing : CMTime? = CMTime(seconds: 0.3, preferredTimescale: 600),
        eColor: Color = .accentColor,
        eFontSize: CGFloat = 17.0,
        command : Binding<PlaybackCommand> = .constant(.play)
    ) {
        self._command = command

        func description(@SettingsBuilder content: () -> [Setting]) -> [Setting] {
          return content()
        }
        
        let settings: VideoSettings = VideoSettings {
            SourceName(fileName)
            Ext(ext)
            Gravity(gravity)
            if let timePublishing{
                timePublishing
           }
            ErrorGroup {
                EColor(eColor)
                EFontSize(eFontSize)
            }
        }
        
        _settings = .constant(settings)
    }
    
    /// Player initializer in a declarative way
    /// - Parameters:
    ///   - settings: Set of settings
    ///   - command: A binding to control playback actions
    public init(
        _ settings: () -> VideoSettings,
        command: Binding<PlaybackCommand> = .constant(.play)
    ) {

        self._command = command
        _settings = .constant(settings())
    }
    
    /// Player initializer in a declarative way
    /// - Parameters:
    ///   - settings: A binding to the set of settings for the video player
    ///   - command: A binding to control playback actions
    public init(
        settings: Binding<VideoSettings>,
        command: Binding<PlaybackCommand> = .constant(.play)
    ) {
        self._settings = settings
        self._command = command
    }
    
    // MARK: - API
       
   /// The body property defines the view hierarchy for the user interface.
   public var body: some View {
       
       
       VStack(spacing: 16) {
//           if let processedImage {
//               Image(nsImage: processedImage)
//           }
           
           GeometryReader { proxy in
               LoopPlayerMultiPlatform(
                settings: $settings,
                command: $command,
                timePublisher: timePublisher,
                snapshotPublisher: snapshotPublisher,
                eventPublisher: eventPublisher
               )
                   .frame(maxWidth: .infinity, maxHeight: .infinity)
                   .onReceive(timePublisher, perform: { time in
                       currentTime = time
                   })
                   .onReceive(snapshotPublisher, perform: { nsImage in
                       let (res, image) = predict(nsImage: nsImage)
                       processedImage = image
                       isAligned = res >= 1 ? true : false
                   })
                   .onReceive(eventPublisher.collect(.byTime(DispatchQueue.main, .seconds(1))), perform: { event in
                       playerEvent = event
                   })
                   .preference(key: CurrentTimePreferenceKey.self, value: currentTime)
                   .preference(key: PlayerEventPreferenceKey.self, value: playerEvent)
                   .cornerRadius(20)
                   .overlay(
                       RoundedRectangle(cornerRadius: 20)
                           .stroke(isAligned ? .green : .red, lineWidth: 8)
                   )
                   .padding(.horizontal, proxy.size.width / 5)
           }
           
           Text(isAligned ? "ALIGNED" : "NOT ALIGNED")
               .font(.system(size: 20, weight: .semibold))
               .foregroundColor(isAligned ? .green : .red)
       }
       
//       if let nsImage {
//           Image(nsImage: nsImage)
//       }
         
   }
}
