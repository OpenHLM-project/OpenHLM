#include <asio.hpp>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

class TCPException : public std::exception {
private:
  std::string message;

public:
  TCPException(const std::string &msg) : message(msg) {}
  const char *what() const noexcept override { return message.c_str(); }
};

class TCPClient {
private:
  asio::io_context io_context;
  asio::ip::tcp::socket socket;
  std::string server_ip;
  int server_port;
  std::atomic<bool> connected;
  std::thread io_thread;

public:
  TCPClient(const std::string &ip, int port)
      : socket(io_context), server_ip(ip), server_port(port), connected(false) {
  }

  ~TCPClient() { disconnect(); }

  bool connect() {
    if (connected)
      return true;

    try {
      asio::ip::tcp::resolver resolver(io_context);
      auto endpoints = resolver.resolve(server_ip, std::to_string(server_port));

      asio::connect(socket, endpoints);
      connected = true;

      // Start io_context in separate thread
      io_thread = std::thread([this]() { io_context.run(); });

      std::cout << "Connected to server " << server_ip << ":" << server_port
                << std::endl;
      return true;
    } catch (const std::exception &e) {
      throw TCPException("Connection failed: " + std::string(e.what()));
    }
  }

  void disconnect() {
    connected = false;

    if (socket.is_open()) {
      std::error_code ec;
      socket.close(ec);
    }

    io_context.stop();

    if (io_thread.joinable()) {
      io_thread.join();
    }

    io_context.restart();
  }

  bool isConnected() const { return connected && socket.is_open(); }

  void sendData(const char *data, uint32_t size) {
    if (!connected || !socket.is_open()) {
      throw TCPException("Not connected to server");
    }

    if (!data || size == 0) {
      throw TCPException("Invalid data or size");
    }

    try {
      asio::write(socket, asio::buffer(data, size));
    } catch (const std::exception &e) {
      connected = false;
      throw TCPException("Send failed: " + std::string(e.what()));
    }
  }

  void sendData(const std::vector<uint8_t> &data) {
    sendData(reinterpret_cast<const char *>(data.data()),
             static_cast<uint32_t>(data.size()));
  }
};

class TCPServer {
private:
  asio::io_context io_context;
  asio::ip::tcp::acceptor acceptor;
  std::shared_ptr<asio::ip::tcp::socket> client_socket;
  int server_port;
  std::atomic<bool> client_connected;
  std::atomic<bool> server_running;
  std::thread server_thread;
  std::promise<void> exit_signal;
  std::future<void> exit_future;

  std::function<void(const std::string &)> data_callback;
  std::function<void()> disconnect_callback;

  void startAccept() {
    client_socket = std::make_shared<asio::ip::tcp::socket>(io_context);

    acceptor.async_accept(
        *client_socket, [this](std::error_code error) { handleAccept(error); });
  }

  void handleAccept(const std::error_code &error) {
    if (!error && server_running) {
      client_connected = true;

      auto endpoint = client_socket->remote_endpoint();
      std::cout << "Client connected from IP: "
                << endpoint.address().to_string()
                << ", Port: " << endpoint.port() << std::endl;

      startReceive();
    } else if (server_running) {
      std::cerr << "Accept error: " << error.message() << std::endl;
      startAccept(); // Continue accepting connections
    }
  }

  void startReceive() {
    if (!client_socket || !client_connected)
      return;

    auto buffer = std::make_shared<std::array<char, 1024>>();

    client_socket->async_read_some(
        asio::buffer(*buffer),
        [this, buffer](std::error_code error, std::size_t bytes_transferred) {
          handleReceive(error, bytes_transferred, buffer);
        });
  }

  void handleReceive(const std::error_code &error,
                     std::size_t bytes_transferred,
                     std::shared_ptr<std::array<char, 1024>> buffer) {
    if (!error && client_connected) {
      std::string received_data(buffer->data(), bytes_transferred);

      if (!received_data.empty() && data_callback) {
        std::cout << "Received from client: " << received_data << std::endl;
        data_callback(received_data);
      }

      startReceive(); // Continue receiving
    } else {
      if (error == asio::error::eof) {
        std::cout << "Client disconnected gracefully" << std::endl;
      } else if (error) {
        std::cerr << "Receive error: " << error.message() << std::endl;
      }

      if (disconnect_callback) {
        disconnect_callback();
      }

      disconnectClient();

      if (server_running) {
        startAccept(); // Wait for new connections
      }
    }
  }

  void serverLoop() {
    try {
      std::cout << "Starting server loop..." << std::endl;
      startAccept();
      io_context.run();
    } catch (const std::exception &e) {
      std::cerr << "Server loop error: " << e.what() << std::endl;
    }

    // Notify main thread we're done
    try {
      exit_signal.set_value();
    } catch (...) {
      // Avoid exception if already set
    }
  }

public:
  TCPServer(const std::string &address)
      : acceptor(io_context), client_connected(false), server_running(false),
        exit_future(exit_signal.get_future()) {
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
    if (server_running) {
      throw TCPException("Server already running");
    }

    try {
      asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), server_port);
      acceptor.open(endpoint.protocol());
      acceptor.set_option(asio::ip::tcp::acceptor::reuse_address(true));
      acceptor.bind(endpoint);
      acceptor.listen();

      server_running = true;
      server_thread = std::thread(&TCPServer::serverLoop, this);
      std::cout << "Server listening on port " << server_port << std::endl;
    } catch (const std::exception &e) {
      throw TCPException("Server start failed: " + std::string(e.what()));
    }
  }

  bool hasClient() {
    return client_socket && client_socket->is_open() && client_connected;
  }

  void stop() {
    std::cout << "[TCPServer] Stopping server..." << std::endl;
    server_running = false;

    if (acceptor.is_open()) {
      std::error_code ec;
      acceptor.close(ec);
    }

    io_context.stop();

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
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch())
                      .count();
    printf("===LATENCY TEST===[%s]\t%lld ms\n", tag.c_str(), now_ms);
  }

  void disconnectClient() {
    if (client_socket && client_socket->is_open()) {
      std::error_code ec;
      client_socket->close(ec);
    }

    if (client_connected) {
      std::cout << "Client disconnected" << std::endl;
    }
    client_connected = false;
    client_socket.reset();
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

/// UDP
class UDPClient {
private:
  asio::io_context io_context;
  asio::ip::udp::socket socket;
  asio::ip::udp::endpoint server_endpoint;
  std::string server_ip;
  int server_port;
  std::atomic<bool> connected;
  std::thread io_thread;

public:
  UDPClient(const std::string &ip, int port)
      : socket(io_context), server_ip(ip), server_port(port), connected(false) {
  }

  ~UDPClient() { disconnect(); }

  bool connect() {
    if (connected)
      return true;

    try {
      asio::ip::udp::resolver resolver(io_context);
      auto endpoints = resolver.resolve(server_ip, std::to_string(server_port));
      server_endpoint = *endpoints.begin();

      socket.open(asio::ip::udp::v4());
      connected = true;

      // Start io_context in separate thread
      io_thread = std::thread([this]() { io_context.run(); });

      std::cout << "UDP client initialized for server " << server_ip << ":"
                << server_port << std::endl;
      return true;
    } catch (const std::exception &e) {
      throw TCPException("UDP connection setup failed: " +
                         std::string(e.what()));
    }
  }

  void disconnect() {
    connected = false;

    if (socket.is_open()) {
      std::error_code ec;
      socket.close(ec);
    }

    io_context.stop();

    if (io_thread.joinable()) {
      io_thread.join();
    }

    io_context.restart();
  }

  bool isConnected() const { return connected && socket.is_open(); }

  void sendData(const char *data, uint32_t size) {
    if (!connected || !socket.is_open()) {
      throw TCPException("UDP socket not initialized");
    }

    if (!data || size == 0) {
      throw TCPException("Invalid data or size");
    }

    try {
      // Check if size exceeds UDP maximum payload size
      const uint32_t UDP_MAX_PAYLOAD =
          65507; // 65535 - 8 (UDP header) - 20 (IP header)
      if (size > UDP_MAX_PAYLOAD) {
        std::cerr << "Data size exceeds UDP maximum payload size: " << size
                  << " > " << UDP_MAX_PAYLOAD << std::endl;
        return;
      }

      socket.send_to(asio::buffer(data, size), server_endpoint);
    } catch (const std::exception &e) {
      throw TCPException("UDP send failed: " + std::string(e.what()));
    }
  }

  void sendData(const std::vector<uint8_t> &data) {
    sendData(reinterpret_cast<const char *>(data.data()),
             static_cast<uint32_t>(data.size()));
  }

  // Async version for better performance
  void sendDataAsync(const char *data, uint32_t size) {
    if (!connected || !socket.is_open()) {
      throw TCPException("UDP socket not initialized");
    }

    if (!data || size == 0) {
      throw TCPException("Invalid data or size");
    }

    // Create a shared buffer to keep data alive during async operation
    auto buffer = std::make_shared<std::vector<char>>(data, data + size);

    socket.async_send_to(
        asio::buffer(*buffer), server_endpoint,
        [buffer](std::error_code error, std::size_t bytes_transferred) {
          if (error) {
            std::cerr << "UDP async send error: " << error.message()
                      << std::endl;
          }
          // buffer automatically cleaned up when lambda goes out of scope
        });
  }

  void sendDataAsync(const std::vector<uint8_t> &data) {
    auto buffer = std::make_shared<std::vector<uint8_t>>(data);

    socket.async_send_to(
        asio::buffer(*buffer), server_endpoint,
        [buffer](std::error_code error, std::size_t bytes_transferred) {
          if (error) {
            std::cerr << "UDP async send error: " << error.message()
                      << std::endl;
          }
        });
  }

  // Optional: Add receive functionality for bidirectional communication
  void startReceive(std::function<void(const std::string &)> callback) {
    if (!connected || !socket.is_open()) {
      return;
    }

    auto buffer = std::make_shared<std::array<char, 1024>>();
    auto sender_endpoint = std::make_shared<asio::ip::udp::endpoint>();

    socket.async_receive_from(
        asio::buffer(*buffer), *sender_endpoint,
        [this, buffer, sender_endpoint,
         callback](std::error_code error, std::size_t bytes_transferred) {
          if (!error && connected) {
            std::string received_data(buffer->data(), bytes_transferred);
            if (callback) {
              callback(received_data);
            }
            // Continue receiving
            startReceive(callback);
          } else if (error && error != asio::error::operation_aborted) {
            std::cerr << "UDP receive error: " << error.message() << std::endl;
          }
        });
  }
};

///