from flask import Flask, jsonify, Response
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
            html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Face Door Demo</title>
    <style>
        body {
            font-family: Arial, "Microsoft YaHei", sans-serif;
            background: #f3f4f6;
            margin: 0;
            padding: 40px;
        }

        .card {
            max-width: 520px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        }

        h1 {
            margin-top: 0;
            font-size: 26px;
            color: #111827;
        }

        .row {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #e5e7eb;
            font-size: 16px;
        }

        .row:last-child {
            border-bottom: none;
        }

        .label {
            color: #6b7280;
        }

        .value {
            font-weight: bold;
            color: #111827;
        }

        .ok {
            color: #16a34a;
        }

        .bad {
            color: #dc2626;
        }

        .door-open {
            color: #2563eb;
        }

        .door-closed {
            color: #111827;
        }

        .tips {
            margin-top: 20px;
            font-size: 13px;
            color: #6b7280;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Face Door Demo 状态面板</h1>

        <div class="row">
            <span class="label">门状态</span>
            <span id="door_state" class="value">loading...</span>
        </div>

        <div class="row">
            <span class="label">最近识别人</span>
            <span id="last_name" class="value">loading...</span>
        </div>

        <div class="row">
            <span class="label">识别分数</span>
            <span id="last_score" class="value">loading...</span>
        </div>

        <div class="row">
            <span class="label">是否授权</span>
            <span id="last_authorized" class="value">loading...</span>
        </div>

        <div class="row">
            <span class="label">最近事件时间</span>
            <span id="last_event_time" class="value">loading...</span>
        </div>

        <div class="tips">
            页面每 1 秒自动刷新一次。原始 JSON 接口：<code>/status</code>
        </div>
    </div>

    <script>
        async function refreshStatus() {
            try {
                const response = await fetch("/status");
                const data = await response.json();

                const doorState = document.getElementById("door_state");
                const lastName = document.getElementById("last_name");
                const lastScore = document.getElementById("last_score");
                const lastAuthorized = document.getElementById("last_authorized");
                const lastEventTime = document.getElementById("last_event_time");

                doorState.textContent = data.door_state;
                lastName.textContent = data.last_name;
                lastScore.textContent = data.last_score;
                lastAuthorized.textContent = data.last_authorized;
                lastEventTime.textContent = data.last_event_time;

                doorState.className = "value " + (
                    data.door_state === "opened" ? "door-open" : "door-closed"
                );

                lastAuthorized.className = "value " + (
                    data.last_authorized ? "ok" : "bad"
                );
            } catch (error) {
                console.error("Failed to fetch status:", error);
            }
        }

        refreshStatus();
        setInterval(refreshStatus, 1000);
    </script>
</body>
</html>
"""
            return Response(html, mimetype="text/html")

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
        print(f"[HTTP] Server started: http://{self.host}:{self.port}/")