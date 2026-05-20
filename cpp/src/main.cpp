#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <ctime>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

class StatusStore {
public:
    std::string to_json() {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ostringstream oss;
        oss << "{";
        oss << "\"door_state\":\"" << door_state_ << "\",";
        oss << "\"last_name\":\"" << last_name_ << "\",";
        oss << "\"last_score\":" << last_score_ << ",";
        oss << "\"last_authorized\":" << (last_authorized_ ? "true" : "false") << ",";
        oss << "\"last_event_time\":\"" << last_event_time_ << "\",";
        oss << "\"camera_status\":\"" << camera_status_ << "\",";
        oss << "\"recognition_status\":\"" << recognition_status_ << "\"";
        oss << "}";

        return oss.str();
    }

    void mock_open() {
        std::lock_guard<std::mutex> lock(mutex_);

        door_state_ = "opened";
        last_name_ = "me";
        last_score_ = 0.99;
        last_authorized_ = true;
        last_event_time_ = now_string();
        recognition_status_ = "recognized";
    }

    void mock_unknown() {
        std::lock_guard<std::mutex> lock(mutex_);

        door_state_ = "closed";
        last_name_ = "unknown";
        last_score_ = 0.32;
        last_authorized_ = false;
        last_event_time_ = now_string();
        recognition_status_ = "unknown";
    }

    void mock_reset() {
        std::lock_guard<std::mutex> lock(mutex_);

        door_state_ = "closed";
        last_name_ = "none";
        last_score_ = 0.0;
        last_authorized_ = false;
        last_event_time_ = "none";
        recognition_status_ = "idle";
    }

private:
    static std::string now_string() {
        std::time_t t = std::time(nullptr);
        char buffer[32];

        std::strftime(
            buffer,
            sizeof(buffer),
            "%Y-%m-%d %H:%M:%S",
            std::localtime(&t)
        );

        return std::string(buffer);
    }

private:
    std::mutex mutex_;

    std::string door_state_ = "closed";
    std::string last_name_ = "none";
    double last_score_ = 0.0;
    bool last_authorized_ = false;
    std::string last_event_time_ = "none";
    std::string camera_status_ = "mock";
    std::string recognition_status_ = "idle";
};

std::string make_http_response(
    const std::string& body,
    const std::string& content_type = "application/json"
) {
    std::ostringstream oss;

    oss << "HTTP/1.1 200 OK\r\n";
    oss << "Content-Type: " << content_type << "\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Connection: close\r\n";
    oss << "\r\n";
    oss << body;

    return oss.str();
}

std::string make_not_found_response() {
    const std::string body = "{\"error\":\"not found\"}";

    std::ostringstream oss;
    oss << "HTTP/1.1 404 Not Found\r\n";
    oss << "Content-Type: application/json\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Connection: close\r\n";
    oss << "\r\n";
    oss << body;

    return oss.str();
}

std::string extract_path(const std::string& request) {
    std::istringstream iss(request);

    std::string method;
    std::string path;
    std::string version;

    iss >> method >> path >> version;

    if (method != "GET") {
        return "";
    }

    return path;
}

int main() {
    constexpr int port = 8080;

    StatusStore status_store;

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "[ERROR] socket() failed\n";
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "[ERROR] bind() failed. Is port already in use?\n";
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 16) < 0) {
        std::cerr << "[ERROR] listen() failed\n";
        close(server_fd);
        return 1;
    }

    std::cout << "[HTTP] C++ server started: http://0.0.0.0:" << port << "/\n";

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(
            server_fd,
            reinterpret_cast<sockaddr*>(&client_addr),
            &client_len
        );

        if (client_fd < 0) {
            std::cerr << "[WARN] accept() failed\n";
            continue;
        }

        char buffer[4096];
        std::memset(buffer, 0, sizeof(buffer));

        ssize_t received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        if (received <= 0) {
            close(client_fd);
            continue;
        }

        std::string request(buffer);
        std::string path = extract_path(request);

        std::string response;

        if (path == "/" || path == "/status") {
            response = make_http_response(status_store.to_json());
        } else if (path == "/mock_open") {
            status_store.mock_open();
            response = make_http_response(status_store.to_json());
        } else if (path == "/mock_unknown") {
            status_store.mock_unknown();
            response = make_http_response(status_store.to_json());
        } else if (path == "/mock_reset") {
            status_store.mock_reset();
            response = make_http_response(status_store.to_json());
        } else {
            response = make_not_found_response();
        }

        send(client_fd, response.c_str(), response.size(), 0);
        close(client_fd);
    }

    close(server_fd);
    return 0;
}
