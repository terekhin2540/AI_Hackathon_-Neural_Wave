//
//  View+.swift
//
//
//  Created by Igor Shelopaev on 09.08.24.
//

import AppKit

internal extension NSView {
    
    /// Finds the first subview of the specified type within the view's direct children.
    func findFirstSubview<T: NSView>(ofType type: T.Type) -> T? {
        return subviews.compactMap { $0 as? T }.first
    }
}
