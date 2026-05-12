from flask import Flask, jsonify
import threading

class HttpServer:
    def __init__(self, status_store, host="0.0.0.0", port=8000):
        self.status_store = status_store
        self.host = host
        self.port = port

        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/status", methods=["GET"])
        def status():
            return jsonify(self.status_store.to_dict())

        @self.app.route("/", methods=["GET"])
        def index():
            return "Face Door Demo is running. Visit /status\n"

    def start(self):
        thread = threading.Thread(
            target=self.app.run,
            kwargs={
                "host": self.host,
                "port": self.port,
                "debug": False,
                "use_reloader": False,
            },
            daemon=True
        )

        thread.start()
        print(f"[HTTP] Server started: http://{self.host}:{self.port}/status")
