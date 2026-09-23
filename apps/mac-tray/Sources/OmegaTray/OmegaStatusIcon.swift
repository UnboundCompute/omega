import AppKit

enum OmegaStatusIcon {
    static func make() -> NSImage {
        let image = NSImage(size: NSSize(width: 18, height: 18), flipped: false) { bounds in
            NSColor.labelColor.setStroke()

            let arc = NSBezierPath()
            arc.lineWidth = 1.7
            arc.lineCapStyle = .round
            arc.move(to: NSPoint(x: bounds.minX + 3.5, y: bounds.maxY - 6))
            arc.curve(
                to: NSPoint(x: bounds.midX, y: bounds.minY + 4.5),
                controlPoint1: NSPoint(x: bounds.minX + 3.5, y: bounds.minY + 7),
                controlPoint2: NSPoint(x: bounds.midX - 2.5, y: bounds.minY + 4.5)
            )
            arc.curve(
                to: NSPoint(x: bounds.maxX - 3.5, y: bounds.maxY - 6),
                controlPoint1: NSPoint(x: bounds.midX + 2.5, y: bounds.minY + 4.5),
                controlPoint2: NSPoint(x: bounds.maxX - 3.5, y: bounds.minY + 7)
            )
            arc.stroke()

            let base = NSBezierPath()
            base.lineWidth = 1.4
            base.lineCapStyle = .round
            base.move(to: NSPoint(x: bounds.minX + 5.5, y: bounds.minY + 3))
            base.line(to: NSPoint(x: bounds.maxX - 5.5, y: bounds.minY + 3))
            base.stroke()
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = "omega"
        return image
    }
}
