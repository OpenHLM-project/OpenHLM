#include <arpa/inet.h>
#include <atomic>
#include <functional>
#include <future>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>

class TCPException : public std::exception {
private:
  std::string message;

public:
  TCPException(const std::string &msg) : message(msg) {}
  const char *what() const noexcept override { return message.c_str(); }
};

class TCPClient {
private:
  int client_socket;
  std::string server_ip;
  int server_port;
  bool connected;

public:
  TCPClient(const std::string &ip, int port)
      : client_socket(-1), server_ip(ip), server_port(port), connected(false) {}

  ~TCPClient() { disconnect(); }

  bool connect() {
    if (connected)
      return true;

    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
      throw TCPException("Socket creation failed: " +
                         std::string(strerror(errno)));
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);

    if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
      close(client_socket);
      client_socket = -1;
      throw TCPException("Invalid server IP address: " + server_ip);
    }

    if (::connect(client_socket, (struct sockaddr *)&server_addr,
                  sizeof(server_addr)) < 0) {
      int error = errno;
      close(client_socket);
      client_socket = -1;
      throw TCPException("Connection failed: " + std::string(strerror(error)));
    }

    connected = true;
    std::cout << "Connected to server " << server_ip << ":" << server_port
              << std::endl;
    return true;
  }

  void disconnect() {
    if (client_socket >= 0) {
      close(client_socket);
      client_socket = -1;
    }
    connected = false;
  }

  bool isConnected() const { return connected; }

  void sendData(const char *data, uint32_t size) {
    if (!connected) {
      throw TCPException("Not connected to server");
    }

    if (!data || size == 0) {
      throw TCPException("Invalid data or size");
    }

    uint32_t total_sent = 0;
    while (total_sent < size) {
      ssize_t sent = send(client_socket, data + total_sent, size - total_sent,
                          MSG_NOSIGNAL);
      if (sent < 0) {
        int error = errno;
        if (error == ECONNRESET || error == EPIPE) {
          connected = false;
          throw TCPException("Connection lost: " +
                             std::string(strerror(error)));
        }
        throw TCPException("Send failed: " + std::string(strerror(error)));
      }
      total_sent += sent;
    }
  }

  void sendData(const std::vector<uint8_t> &data) {
    sendData(reinterpret_cast<const char *>(data.data()),
             static_cast<uint32_t>(data.size()));
  }
};

class TCPServer {
private:
  int server_socket;
  int client_socket;
  int server_port;
  std::atomic<bool> client_connected;
  std::atomic<bool> server_running;
  std::thread server_thread;
  std::promise<void> exit_signal;
  std::future<void> exit_future;

  std::function<void(const std::string &)> data_callback;
  std::function<void()> disconnect_callback;

  void serverLoop() {
    while (server_running) {
      try {
        std::cout << "Waiting for a new client connection..." << std::endl;
        acceptConnection();
        handleClient();
      } catch (const TCPException &e) {
        std::cerr << "Error in server loop: " << e.what() << std::endl;
      }
      disconnectClient();
    }

    // Notify main thread we're done
    try {
      exit_signal.set_value();
    } catch (...) {
      // Avoid exception if already set
    }
  }

  void handleClient() {
    std::cout << "Handling client communication..." << std::endl;
    std::vector<uint8_t> buffer(1024);
    while (client_connected) {
      ssize_t received = recv(client_socket, buffer.data(), buffer.size(), 0);
      if (received < 0) {
        int error = errno;
        if (error == ECONNRESET || error == EPIPE) {
          std::cerr << "Connection lost: " << strerror(error) << std::endl;
          break;
        }
        std::cerr << "Receive failed: " << strerror(error) << std::endl;
        break;
      } else if (received == 0) {
        std::cout << "Client disconnected gracefully" << std::endl;
        if (disconnect_callback) {
          disconnect_callback();
        }
        break;
      }

      if (received > 0 && data_callback) {
        std::vector<uint8_t> data(buffer.begin(), buffer.begin() + received);
        std::string binary_data(data.begin(), data.end());
        std::cout << "Received " << received << " bytes from client"
                  << std::endl;
        data_callback(binary_data);
      }
    }
  }

public:
  TCPServer(const std::string &address)
      : server_socket(-1), client_socket(-1), client_connected(false),
        server_running(false), exit_future(exit_signal.get_future()) {
    size_t colon_pos = address.find(':');
    if (colon_pos == std::string::npos) {
      throw TCPException("Invalid address format. Expected 'ip:port'");
    }

    std::string port_str = address.substr(colon_pos + 1);
    try {
      server_port = std::stoi(port_str);
    } catch (...) {
      throw TCPException("Invalid port number: " + port_str);
    }

    if (server_port <= 0 || server_port > 65535) {
      throw TCPException("Port number out of range");
    }

    std::cout << "Server initialized with Port: " << server_port << std::endl;
  }

  void start() {
    if (server_socket >= 0) {
      throw TCPException("Server already running");
    }

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
      throw TCPException("Socket creation failed: " +
                         std::string(strerror(errno)));
    }

    // Enable address reuse
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) <
        0) {
      close(server_socket);
      server_socket = -1;
      throw TCPException("setsockopt(SO_REUSEADDR) failed: " +
                         std::string(strerror(errno)));
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(server_port);

    if (bind(server_socket, (struct sockaddr *)&server_addr,
             sizeof(server_addr)) < 0) {
      close(server_socket);
      server_socket = -1;
      throw TCPException("Bind failed: " + std::string(strerror(errno)));
    }

    if (listen(server_socket, 1) < 0) {
      close(server_socket);
      server_socket = -1;
      throw TCPException("Listen failed: " + std::string(strerror(errno)));
    }

    server_running = true;
    server_thread = std::thread(&TCPServer::serverLoop, this);
    std::cout << "Server listening on port " << server_port << std::endl;
  }

  bool hasClient() { return client_socket != -1; }

  void TestLoopLatency() {
    if (client_socket != -1) {
      std::string msg = "LOOPTEST";
      size_t size = msg.size();

      std::vector<uint8_t> packet(4 + size);
      packet[0] = (size >> 24) & 0xFF;
      packet[1] = (size >> 16) & 0xFF;
      packet[2] = (size >> 8) & 0xFF;
      packet[3] = (size)&0xFF;
      std::copy(msg.begin(), msg.end(), packet.begin() + 4);

      send(client_socket, packet.data(), packet.size(), 0);

      std::cout << "[TCPServer] Sent loop test msg to client." << std::endl;
    }
  }

  void stop() {
    std::cout << "[TCPServer] Stopping server..." << std::endl;
    server_running = false;

    if (server_socket >= 0) {
      close(server_socket);
      server_socket = -1;
    }

    if (exit_future.valid()) {
      auto status = exit_future.wait_for(std::chrono::seconds(1));
      if (status == std::future_status::timeout) {
        std::cerr << "Server thread timeout. Detaching...\n";
        if (server_thread.joinable())
          server_thread.detach();
      } else {
        if (server_thread.joinable())
          server_thread.join();
      }
    }

    disconnectClient();
    std::cout << "Server stopped" << std::endl;
  }

  ~TCPServer() { stop(); }

  void inline printTimeMs(std::string tag) {
    auto now = std::chrono::system_clock::now();
    // point the current time in milliseconds
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch())
                      .count();
    printf("===LATENCY TEST===[%s]\t%lld ms\n", tag.c_str(),
           static_cast<long long>(now_ms));
  }

  bool acceptConnection() {
    if (client_connected) {
      throw TCPException("Client already connected");
    }

    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    client_socket =
        accept(server_socket, (struct sockaddr *)&client_addr, &client_len);
    if (client_socket < 0) {
      throw TCPException("Accept failed: " + std::string(strerror(errno)));
    }

    client_connected = true;
    std::cout << "Client connected from IP: " << inet_ntoa(client_addr.sin_addr)
              << ", Port: " << ntohs(client_addr.sin_port) << std::endl;

    // // loop test
    // // Sleep for 1 second
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // printTimeMs("Loop Send");
    // TestLoopLatency();

    return true;
  }

  void disconnectClient() {
    if (client_socket >= 0) {
      close(client_socket);
      client_socket = -1;
    }
    if (client_connected) {
      std::cout << "Client disconnected" << std::endl;
    }
    client_connected = false;
  }

  bool isClientConnected() const { return client_connected; }

  void
  setDataCallback(const std::function<void(const std::string &)> &callback) {
    data_callback = callback;
  }

  void setDisconnectCallback(const std::function<void()> &callback) {
    disconnect_callback = callback;
  }
};