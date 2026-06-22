import Cocoa
import ServiceManagement

enum ServerState {
    case stopped, starting, running
}

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

    private let host = "0.0.0.0"
    private let port = "8000"
    private var healthURL: URL { URL(string: "http://127.0.0.1:\(port)/health")! }
    private var docsURL: URL { URL(string: "http://127.0.0.1:\(port)/docs")! }

    private var ttsDir: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("projects/tts-2")
    }
    private var pythonURL: URL { ttsDir.appendingPathComponent(".venv-mlx/bin/python") }
    private var scriptURL: URL { ttsDir.appendingPathComponent("server.py") }

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
        let docs = NSMenuItem(title: "Open Web UI", action: #selector(openDocs), keyEquivalent: "")
        docs.target = self
        menu.addItem(docs)
        menu.addItem(loginItem)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Chatterbox TTS", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        statusItem.menu = menu
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
            process.environment = env
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
        let task = URLSession.shared.dataTask(with: request) { _, response, _ in
            let up = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                if up {
                    self.state = .running
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
        let symbol: String
        switch state {
        case .running: symbol = "waveform"
        case .starting: symbol = "waveform.badge.exclamationmark"
        case .stopped: symbol = "waveform.slash"
        }
        let image = NSImage(systemSymbolName: symbol, accessibilityDescription: "Chatterbox TTS")
        image?.isTemplate = true
        statusItem.button?.image = image

        switch state {
        case .running: statusMenuItem.title = "Chatterbox TTS · Running"
        case .starting: statusMenuItem.title = "Chatterbox TTS · Starting…"
        case .stopped: statusMenuItem.title = "Chatterbox TTS · Stopped"
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
