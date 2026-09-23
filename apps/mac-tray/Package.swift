// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "OmegaTray",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "OmegaTray", targets: ["OmegaTray"])
    ],
    targets: [
        .executableTarget(
            name: "OmegaTray",
            path: "Sources/OmegaTray"
        ),
        .testTarget(
            name: "OmegaTrayTests",
            dependencies: ["OmegaTray"],
            path: "Tests/OmegaTrayTests"
        )
    ]
)
