//
//  Utils.swift
//  NeuraWave
//
//  Created by Roman Rakhlin on 10/27/24.
//

import CoreML
import AppKit
import CoreImage

func predict(nsImage: NSImage) -> (Float, NSImage?) {
    let model = try! defergo()
    
    let imageProcessor = ImageProcessor(targetSize: 384)
    let processedImage = imageProcessor.processImage(nsImage)

    do {
        if let processedImage, let multiArray = nsImageToMLMultiArray(processedImage, size: processedImage.size) {
            let prediction = try model.prediction(x_1: multiArray)
            let floatValue = prediction.var_2423[0].floatValue
            let probability = 1 / (1 + exp(-floatValue))
            print("Real probability: \(probability), Predicted probability: \(probability * 100)")
            return (probability * 100, processedImage)
        } else {
            print("kjdfngkjdf")
        }
    } catch {
        print("kdjfngfd", error)
    }
    
    return (0, nil)
}

class ImageProcessor {
    var targetSize: Int
    var preprocess: String
    var maskLeft: Bool

    init(targetSize: Int, preprocess: String = "grayscale", maskLeft: Bool = false) {
        self.targetSize = targetSize
        self.preprocess = preprocess
        self.maskLeft = maskLeft
    }

    func processImage(_ nsImage: NSImage) -> NSImage? {
        guard let image = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        
        let imageArray = preprocessImage(image: image)
        let normalizedImageArray = normalize(imageArray)

        var imageCI = CIImage(cgImage: normalizedImageArray)

        if maskLeft {
            imageCI = maskFromTheLeft(image: imageCI)
        }

        // Convert CIImage back to NSImage
        return convertToNSImage(imageCI)
    }

    private func preprocessImage(image: CGImage) -> CGImage {
        let width = image.width
        let height = image.height
        let cropWidth = width - height
        
        // Crop the image to square
        guard let croppedImage = image.cropping(to: CGRect(x: cropWidth, y: 0, width: height, height: height)) else {
            return image // return original if cropping fails
        }
        
        // Resize the image
        let resizedImage = resizeImage(croppedImage, to: CGSize(width: targetSize, height: targetSize))

        // Convert to grayscale if requested
        if preprocess == "grayscale" {
            return convertToGrayscale(resizedImage)
        } else if preprocess == "edges" {
            return preprocessDark(resizedImage)
        }

        return resizedImage
    }

    private func resizeImage(_ image: CGImage, to size: CGSize) -> CGImage {
        let context = CGContext(data: nil,
                                width: Int(size.width),
                                height: Int(size.height),
                                bitsPerComponent: 8,
                                bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        
        return context.makeImage()!
    }

    private func convertToGrayscale(_ image: CGImage) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let context = CGContext(data: nil,
                                width: image.width,
                                height: image.height,
                                bitsPerComponent: 8,
                                bytesPerRow: 0,
                                space: colorSpace,
                                bitmapInfo: 0)!
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        
        return context.makeImage()!
    }

    private func normalize(_ image: CGImage) -> CGImage {
        // Normalize the pixel values (0 to 255 to 0 to 1)
        // Here you might need to implement the pixel normalization based on your requirements
        return image // Return unchanged for simplicity; normalization would be done later in the pipeline
    }

    private func preprocessDark(_ image: CGImage) -> CGImage {
        // Convert to grayscale
        let grayImage = convertToGrayscale(image)
        
        // Get pixel data
        let context = CIContext()
        let ciImage = CIImage(cgImage: grayImage)
        guard let pixelBuffer = context.createCGImage(ciImage, from: ciImage.extent) else { return image }
        
        let brightness: Float
        let contrast: Float
        let meanBrightness = calculateMeanBrightness(grayImage)

        if meanBrightness < 55 { // Dark
            brightness = 155
            contrast = 1.5
        } else { // Light
            brightness = 75
            contrast = 1.35
        }

        let adjustedImage = applyBrightnessContrast(to: pixelBuffer, brightness: brightness, contrast: contrast)
        return adjustedImage
    }

    private func calculateMeanBrightness(_ image: CGImage) -> Float {
        // Implement your logic to calculate mean brightness from the CGImage
        return 0.0 // Placeholder
    }

    private func applyBrightnessContrast(to image: CGImage, brightness: Float, contrast: Float) -> CGImage {
        // Implement your logic to adjust brightness and contrast using Core Image filters
        return image // Placeholder
    }

    private func maskFromTheLeft(image: CIImage) -> CIImage {
        // Create a mask to black out the left half of the image
        let maskFilter = CIFilter(name: "CICrop")!
        maskFilter.setValue(image, forKey: kCIInputImageKey)
        maskFilter.setValue(CIVector(x: image.extent.midX, y: image.extent.midY), forKey: "inputBottomRight")
        maskFilter.setValue(CIVector(x: image.extent.midX / 2, y: image.extent.height), forKey: "inputTopLeft")
        
        return maskFilter.outputImage ?? image
    }

    private func convertToNSImage(_ ciImage: CIImage) -> NSImage? {
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return NSImage(cgImage: cgImage, size: NSSize(width: ciImage.extent.width, height: ciImage.extent.height))
    }
}


func nsImageToMLMultiArray(_ image: NSImage, size: CGSize) -> MLMultiArray? {
    // 1. Convert NSImage to CGImage
    guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("Could not convert NSImage to CGImage")
        return nil
    }
    
    // 2. Resize and normalize the image (if needed)
    let width = Int(size.width)
    let height = Int(size.height)
    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        print("Could not create graphics context")
        return nil
    }
    context.draw(cgImage, in: CGRect(origin: .zero, size: size))
    
    guard let pixelBuffer = context.data else {
        print("Could not get pixel buffer from context")
        return nil
    }
    
    // 3. Create MLMultiArray to hold pixel data
    let arrayShape = [1, 3, height, width] as [NSNumber]  // Updated shape to match model
    guard let mlArray = try? MLMultiArray(shape: arrayShape, dataType: .float32) else {
        print("Could not create MLMultiArray")
        return nil
    }
    
    // 4. Fill MLMultiArray with normalized pixel data
    let bytePtr = pixelBuffer.bindMemory(to: UInt8.self, capacity: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let pixelIndex = (y * width + x) * 4
            let r = Float(bytePtr[pixelIndex]) / 255.0
            let g = Float(bytePtr[pixelIndex + 1]) / 255.0
            let b = Float(bytePtr[pixelIndex + 2]) / 255.0
            
            // Populate the MLMultiArray in the required shape
            mlArray[[0, 0, y as NSNumber, x as NSNumber]] = NSNumber(value: r)
            mlArray[[0, 1, y as NSNumber, x as NSNumber]] = NSNumber(value: g)
            mlArray[[0, 2, y as NSNumber, x as NSNumber]] = NSNumber(value: b)
        }
    }
    
    return mlArray
}
