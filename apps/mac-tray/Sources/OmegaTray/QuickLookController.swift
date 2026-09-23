import AppKit
import QuickLookUI

@MainActor
final class QuickLookController: NSObject, @preconcurrency QLPreviewPanelDataSource {
    static let shared = QuickLookController()

    private var items: [URL] = []

    func preview(_ urls: [URL], startingWith selectedURL: URL) {
        guard !urls.isEmpty, let panel = QLPreviewPanel.shared() else { return }

        items = urls
        panel.dataSource = self
        panel.reloadData()
        panel.currentPreviewItemIndex = urls.firstIndex(of: selectedURL) ?? 0
        panel.makeKeyAndOrderFront(nil)
    }

    func numberOfPreviewItems(in panel: QLPreviewPanel!) -> Int {
        items.count
    }

    func previewPanel(_ panel: QLPreviewPanel!, previewItemAt index: Int) -> (any QLPreviewItem)! {
        items[index] as NSURL
    }
}
