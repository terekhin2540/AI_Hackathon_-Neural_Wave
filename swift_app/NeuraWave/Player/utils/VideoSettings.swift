//
//  Settings.swift
//  
//
//  Created by Igor Shelopaev on 07.07.2023.
//

import SwiftUI
import AVKit

/// Represents a structure for video settings.
@available(iOS 14.0, macOS 11.0, tvOS 14.0, *)
public struct VideoSettings: Equatable{
    
    // MARK: - Public properties
    
    /// Name of the video to play
    public let name: String
    
    /// Video extension
    public let ext: String
    
    /// Loop video
    public let loop: Bool
    
    /// Mute video
    public let mute: Bool
    
    /// Don't auto play video after initialization
    public let notAutoPlay: Bool
    
    /// A CMTime value representing the interval at which the player's current time should be published.
    /// If set, the player will publish periodic time updates based on this interval.
    public let timePublishing: CMTime?
    
    public let snapshotPublishing: CMTime?
           
    /// A structure that defines how a layer displays a player’s visual content within the layer’s bounds
    public let gravity: AVLayerVideoGravity
    
    /// Error message text color
    public let errorColor : Color
        
    /// Size of the error text Default : 17.0
    public let errorFontSize : CGFloat
    
    /// Do not show inner error showcase component
    public let errorWidgetOff: Bool
        
    /// Are the params unique
    public var areUnique : Bool {
        unique
    }
    
    // MARK: - Private properties
    
    /// Is settings are unique
    private let unique : Bool

    // MARK: - Life circle
    
    /// Initializes a new instance of `VideoSettings` with specified values for various video properties.
    /// - Parameters:
    ///   - name: The name of the video.
    ///   - ext: The video file extension.
    ///   - loop: A Boolean indicating whether the video should loop.
    ///   - mute: A Boolean indicating whether the video should be muted.
    ///   - notAutoPlay: A Boolean indicating whether the video should not auto-play.
    ///   - timePublishing: A `CMTime` value representing the interval for time publishing updates, or `nil`.
    ///   - gravity: The `AVLayerVideoGravity` value defining how the video should be displayed in its layer.
    ///   - errorColor: The color used for error messages.
    ///   - errorFontSize: The font size for error messages.
    ///   - errorWidgetOff: A Boolean indicating whether the error widget should be turned off.
    public init(name: String, ext: String, loop: Bool, mute: Bool, notAutoPlay: Bool, timePublishing: CMTime?, snapshotPublishing: CMTime?, gravity: AVLayerVideoGravity, errorColor: Color, errorFontSize: CGFloat, errorWidgetOff: Bool) {
        self.name = name
        self.ext = ext
        self.loop = loop
        self.mute = mute
        self.notAutoPlay = notAutoPlay
        self.timePublishing = timePublishing
        self.snapshotPublishing = snapshotPublishing
        self.gravity = gravity
        self.errorColor = errorColor
        self.errorFontSize = errorFontSize
        self.errorWidgetOff = errorWidgetOff
        self.unique = true
    }
        
    /// Initializes `VideoSettings` using a settings builder closure.
    /// - Parameter builder: A block builder that generates an array of settings.
    public init(@SettingsBuilder builder: () -> [Setting]){
        let settings = builder()
        
        unique = check(settings)
        
        name = settings.fetch(by : "name", defaulted: "")
        
        ext = settings.fetch(by : "ext", defaulted: "mp4")
        
        gravity = settings.fetch(by : "gravity", defaulted: .resizeAspect)
        
        errorColor = settings.fetch(by : "errorColor", defaulted: .red)
        
        errorFontSize = settings.fetch(by : "errorFontSize", defaulted: 17)
        
        timePublishing = settings.fetch(by : "timePublishing", defaulted: nil)
        
        snapshotPublishing = settings.fetch(by : "snapshotPublishing", defaulted: nil)
        
        loop = settings.contains(.loop)
        
        mute = settings.contains(.mute)
        
        notAutoPlay = settings.contains(.notAutoPlay)
        
        errorWidgetOff = settings.contains(.errorWidgetOff)
    }
}

@available(iOS 14.0, macOS 11.0, tvOS 14.0, *)
public extension VideoSettings {
   
    /// Returns a new instance of VideoSettings with loop set to false and notAutoPlay set to true, keeping other settings unchanged.
    var GetSettingsWithNotAutoPlay : VideoSettings {
        VideoSettings(name: self.name, ext: self.ext, loop: self.loop, mute: self.mute, notAutoPlay: true, timePublishing: self.timePublishing, snapshotPublishing: self.snapshotPublishing, gravity: self.gravity, errorColor: self.errorColor, errorFontSize: self.errorFontSize, errorWidgetOff: self.errorWidgetOff)
    }
    
    /// Checks if the asset has changed based on the provided settings and current asset.
    /// - Parameters:
    ///   - asset: The current asset being played.
    /// - Returns: A new `AVURLAsset` if the asset has changed, or `nil` if the asset remains the same.
    func getAssetIfDifferent(than asset: AVURLAsset?) -> AVURLAsset?{
        let newAsset =  assetFor(self)
        
        if asset == nil {
            return newAsset
        }
        
        if let newUrl = newAsset?.url, let oldUrl = asset?.url, newUrl != oldUrl{
            return newAsset
        }

        return nil
    }
}

/// Check if unique
/// - Parameter settings: Passed array of settings flatted by block builder
/// - Returns: True - unique False - not
fileprivate func check(_ settings : [Setting]) -> Bool{
    let cases : [String] = settings.map{ $0.caseName }
    let set = Set(cases)
    return cases.count == set.count    
}

