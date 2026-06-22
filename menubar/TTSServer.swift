import Cocoa
import ServiceManagement

enum ServerState {
    case stopped, starting, loading, running
}

struct TTSModel {
    let label: String
    let repo: String
}

let availableModels: [TTSModel] = [
    TTSModel(label: "Chatterbox", repo: "mlx-community/chatterbox-fp16"),
    TTSModel(label: "OpenAudio (Fish S2 Pro)", repo: "mlx-community/fish-audio-s2-pro-bf16"),
    TTSModel(label: "Kokoro", repo: "mlx-community/Kokoro-82M-bf16"),
    TTSModel(label: "Orpheus", repo: "mlx-community/orpheus-3b-0.1-ft-4bit"),
]

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var serverProcess: Process?
    private var state: ServerState = .stopped
    private var healthTimer: Timer?

    private let statusMenuItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    private let startItem = NSMenuItem(title: "Start", action: #selector(start), keyEquivalent: "")
    private let stopItem = NSMenuItem(title: "Stop", action: #selector(stop), keyEquivalent: "")
    private let restartItem = NSMenuItem(title: "Restart", action: #selector(restart), keyEquivalent: "")
    private let loginItem = NSMenuItem(title: "Launch at Login", action: #selector(toggleLogin), keyEquivalent: "")
    private let modelItem = NSMenuItem(title: "Model", action: nil, keyEquivalent: "")

    // Custom menu-bar logo: white waveform bars (template image, system-tinted).
    static let logoImage: NSImage = {
        let size = NSSize(width: 22, height: 18)
        let img = NSImage(size: size, flipped: false) { _ in
            let heights: [CGFloat] = [5.6, 9.8, 14, 9.8, 5.6]
            let barW: CGFloat = 2.6, gap: CGFloat = 1.8
            let total = CGFloat(heights.count) * barW + CGFloat(heights.count - 1) * gap
            var x = (size.width - total) / 2
            let cy = size.height / 2
            NSColor.black.setFill()
            for h in heights {
                let rect = NSRect(x: x, y: cy - h / 2, width: barW, height: h)
                NSBezierPath(roundedRect: rect, xRadius: barW / 2, yRadius: barW / 2).fill()
                x += barW + gap
            }
            return true
        }
        img.isTemplate = true
        return img
    }()

    private let modelDefaultsKey = "selectedModelRepo"
    private var selectedModel: TTSModel {
        let repo = UserDefaults.standard.string(forKey: modelDefaultsKey)
        return availableModels.first { $0.repo == repo } ?? availableModels[0]
    }

    private let host = "0.0.0.0"
    private let port = "8000"
    private var healthURL: URL { URL(string: "http://127.0.0.1:\(port)/health")! }
    private var loadModelURL: URL { URL(string: "http://127.0.0.1:\(port)/v1/models/load")! }
    private var docsURL: URL { URL(string: "http://127.0.0.1:\(port)/docs")! }

    private var ttsDir: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("projects/tts-2")
    }
    private var pythonURL: URL { ttsDir.appendingPathComponent(".venv-mlx/bin/python") }
    private var scriptURL: URL { ttsDir.appendingPathComponent("server.py") }

    private var logDirURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/TTSServer")
    }
    private var logFileURL: URL { logDirURL.appendingPathComponent("server.log") }

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        buildMenu()
        startItem.target = self
        stopItem.target = self
        restartItem.target = self
        loginItem.target = self
        start()
        healthTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            self?.checkHealth()
        }
        render()
    }

    func applicationWillTerminate(_ notification: Notification) {
        terminateServer()
    }

    private func buildMenu() {
        let menu = NSMenu()
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)
        menu.addItem(.separator())
        menu.addItem(startItem)
        menu.addItem(stopItem)
        menu.addItem(restartItem)
        menu.addItem(.separator())
        let modelMenu = NSMenu()
        for (i, m) in availableModels.enumerated() {
            let item = NSMenuItem(title: m.label, action: #selector(selectModel(_:)), keyEquivalent: "")
            item.target = self
            item.tag = i
            modelMenu.addItem(item)
        }
        modelItem.submenu = modelMenu
        menu.addItem(modelItem)
        menu.addItem(.separator())
        let docs = NSMenuItem(title: "Open Web UI", action: #selector(openDocs), keyEquivalent: "")
        docs.target = self
        menu.addItem(docs)
        let logs = NSMenuItem(title: "Open Logs", action: #selector(openLogs), keyEquivalent: "")
        logs.target = self
        menu.addItem(logs)
        menu.addItem(loginItem)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit TTS Server", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        statusItem.menu = menu
    }

    private func openLogHandle() -> FileHandle? {
        let fm = FileManager.default
        try? fm.createDirectory(at: logDirURL, withIntermediateDirectories: true)
        if !fm.fileExists(atPath: logFileURL.path) {
            fm.createFile(atPath: logFileURL.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: logFileURL) else { return nil }
        handle.seekToEndOfFile()
        return handle
    }

    @objc private func openLogs() {
        if !FileManager.default.fileExists(atPath: logFileURL.path) {
            _ = openLogHandle()
        }
        NSWorkspace.shared.open(logFileURL)
    }

    @objc private func start() {
        guard serverProcess == nil else { return }
        checkHealth { [weak self] alreadyUp in
            guard let self else { return }
            if alreadyUp { return }
            let process = Process()
            process.executableURL = self.pythonURL
            process.arguments = [self.scriptURL.path]
            process.currentDirectoryURL = self.ttsDir
            var env = ProcessInfo.processInfo.environment
            env["TTS_HOST"] = self.host
            env["TTS_PORT"] = self.port
            env["TTS_MODEL"] = self.selectedModel.repo
            process.environment = env
            if let logHandle = self.openLogHandle() {
                process.standardOutput = logHandle
                process.standardError = logHandle
            }
            process.terminationHandler = { [weak self] _ in
                DispatchQueue.main.async {
                    self?.serverProcess = nil
                    self?.state = .stopped
                    self?.render()
                }
            }
            do {
                try process.run()
                self.serverProcess = process
                self.state = .starting
            } catch {
                self.state = .stopped
            }
            self.render()
        }
    }

    @objc private func stop() {
        terminateServer()
        state = .stopped
        render()
    }

    @objc private func restart() {
        terminateServer()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
            self?.start()
        }
    }

    private func terminateServer() {
        guard let process = serverProcess else { return }
        process.terminationHandler = nil
        process.terminate()
        serverProcess = nil
    }

    @objc private func selectModel(_ sender: NSMenuItem) {
        let model = availableModels[sender.tag]
        guard model.repo != selectedModel.repo else { return }
        UserDefaults.standard.set(model.repo, forKey: modelDefaultsKey)
        render()
        // Stopped: persist only; the choice is applied via TTS_MODEL on next Start.
        // Running: hot-swap in place via the API (no process restart).
        if state != .stopped {
            loadModel(model.repo)
        }
    }

    private func loadModel(_ repo: String) {
        state = .loading
        render()
        var request = URLRequest(url: loadModelURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["model": repo])
        request.timeoutInterval = 5
        URLSession.shared.dataTask(with: request) { [weak self] _, _, _ in
            DispatchQueue.main.async { self?.checkHealth() }
        }.resume()
    }

    @objc private func openDocs() {
        NSWorkspace.shared.open(docsURL)
    }

    @objc private func quit() {
        terminateServer()
        NSApplication.shared.terminate(nil)
    }

    @objc private func toggleLogin() {
        let service = SMAppService.mainApp
        do {
            if service.status == .enabled {
                try service.unregister()
            } else {
                try service.register()
            }
        } catch {
            NSLog("login toggle failed: \(error)")
        }
        render()
    }

    private func checkHealth(_ completion: ((Bool) -> Void)? = nil) {
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 2
        let task = URLSession.shared.dataTask(with: request) { data, response, _ in
            let up = (response as? HTTPURLResponse)?.statusCode == 200
            var serverState: String?
            if up, let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                serverState = obj["state"] as? String
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                if up {
                    self.state = (serverState == "ready") ? .running : .loading
                } else if self.serverProcess != nil {
                    self.state = .starting
                } else {
                    self.state = .stopped
                }
                self.render()
                completion?(up)
            }
        }
        task.resume()
    }

    private func render() {
        statusItem.button?.image = AppDelegate.logoImage
        switch state {
        case .running: statusItem.button?.alphaValue = 1.0
        case .loading, .starting: statusItem.button?.alphaValue = 0.6
        case .stopped: statusItem.button?.alphaValue = 0.35
        }

        let model = selectedModel
        switch state {
        case .running: statusMenuItem.title = "\(model.label) · Running"
        case .loading: statusMenuItem.title = "\(model.label) · Loading…"
        case .starting: statusMenuItem.title = "\(model.label) · Starting…"
        case .stopped: statusMenuItem.title = "\(model.label) · Stopped"
        }
        modelItem.title = "Model: \(model.label)"
        modelItem.submenu?.items.forEach {
            $0.state = availableModels[$0.tag].repo == model.repo ? .on : .off
        }

        startItem.isEnabled = state == .stopped
        stopItem.isEnabled = state != .stopped
        loginItem.state = SMAppService.mainApp.status == .enabled ? .on : .off
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
