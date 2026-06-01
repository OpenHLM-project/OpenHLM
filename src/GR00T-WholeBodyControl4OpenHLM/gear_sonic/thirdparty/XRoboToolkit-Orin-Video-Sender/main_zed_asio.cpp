#include <chrono>
#include <csignal>
#include <glib-unix.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <string>
#include <thread>
#include <openssl/md5.h>

#include "network_asio.hpp"

std::unique_ptr<TCPClient> sender_ptr;
volatile sig_atomic_t stop_requested = 0;
bool send_enabled = false; // <-- Global flag for sending

// client for sending
std::string send_to_server = "";
int send_to_port = 0;

// server for listening to control command
std::unique_ptr<TCPServer> server_ptr; // Pointer to TCPServer
bool encoding_enabled = false;         // Flag to control encoding and sending

bool initialize_sender() {
  int retry = 10;
  while (retry > 0 && !sender_ptr) {
    try {
      sender_ptr = std::unique_ptr<TCPClient>(
          new TCPClient(send_to_server, send_to_port));
      std::cout << "Attempting to connect to " << send_to_server << ":"
                << send_to_port << std::endl;
      sender_ptr->connect();
      return true;
    } catch (const TCPException &e) {
      std::cerr << "Failed to connect to server: " << e.what() << std::endl;
      sender_ptr = nullptr;
    }
    // Sleep for 1 second
    std::this_thread::sleep_for(std::chrono::seconds(1));
    retry--;
  }
  return false;
}

std::string inline printTimeMs(std::string tag, bool force_print = true) {
  auto now = std::chrono::system_clock::now();
  // point the current time in milliseconds
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch())
                    .count();

  std::stringstream ss;
  ss << "===LATENCY TEST===[" << tag.c_str() << "]\t" << now_ms << " ms\n";
  std::string formatted_string = ss.str();

  if(force_print) {
    std::cout << formatted_string;
  }
  return formatted_string;
}

void onDataCallback(const std::string &command) {
  if (command == "StartRobotCameraStream") {
    // initialize the sender if needed
    if (initialize_sender()) {
      // enable encoding and sending
      encoding_enabled = true;
      send_enabled = true;
    } else {
      // failed to intialized sender, quit
      stop_requested = 1;
      return;
    }
    std::cout << "Started encoding and sending data" << std::endl;
  } else if (command == "StopRobotCameraStream") {
    encoding_enabled = false;
    send_enabled = false;
    std::cout << "Stopped encoding and sending data" << std::endl;
  } else if (command.substr(0, 8) == "LOOPTEST") {
    printTimeMs("Loop Receive");
  } else if (command.substr(0, 12) == "MediaDecoder") {
    printTimeMs("Java - " + command);
  } else {
    std::cerr << "Unknown command received: " << command << std::endl;
  }
}
///

void onDisconnectCallback() {
  // release the send thread
  std::cout << "onDisconnectCallback" << std::endl;
  sender_ptr->~TCPClient();
  sender_ptr = nullptr;
}

void handle_sigint(int) {
  std::cout << "\nSIGINT received. Stopping ..." << std::endl;
  if (server_ptr) {
    server_ptr->stop(); // Gracefully stop the server
    server_ptr = nullptr;
  }
  if (sender_ptr) {
    sender_ptr->disconnect();
    server_ptr = nullptr;
  }
  stop_requested = 1;
}

void printErrorAndQuit(const std::string &error_msg) {
  std::cerr << "Error: " << error_msg << std::endl;
  stop_requested = 1;
}

// Reference:
// https://github.com/stereolabs/zed-sdk/blob/master/object%20detection/birds%20eye%20viewer/cpp/include/utils.hpp#L82-L106
inline cv::Mat slMat2cvMat(sl::Mat &input) {
  // Mapping between MAT_TYPE and CV_TYPE
  int cv_type = -1;
  switch (input.getDataType()) {
  case sl::MAT_TYPE::F32_C1:
    cv_type = CV_32FC1;
    break;
  case sl::MAT_TYPE::F32_C2:
    cv_type = CV_32FC2;
    break;
  case sl::MAT_TYPE::F32_C3:
    cv_type = CV_32FC3;
    break;
  case sl::MAT_TYPE::F32_C4:
    cv_type = CV_32FC4;
    break;
  case sl::MAT_TYPE::U8_C1:
    cv_type = CV_8UC1;
    break;
  case sl::MAT_TYPE::U8_C2:
    cv_type = CV_8UC2;
    break;
  case sl::MAT_TYPE::U8_C3:
    cv_type = CV_8UC3;
    break;
  case sl::MAT_TYPE::U8_C4:
    cv_type = CV_8UC4;
    break;
  default:
    break;
  }

  return cv::Mat(input.getHeight(), input.getWidth(), cv_type,
                 input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

GstFlowReturn on_new_sample(GstAppSink *sink, gpointer user_data) {
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample)
    return GST_FLOW_ERROR;

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    printTimeMs("After Encoding");
    const uint8_t *data = map.data;
    gsize size = map.size;
    if (send_enabled && sender_ptr && sender_ptr->isConnected() && data &&
        size > 0) {
      try {
        std::vector<uint8_t> packet(4 + size);
        packet[0] = (size >> 24) & 0xFF;
        packet[1] = (size >> 16) & 0xFF;
        packet[2] = (size >> 8) & 0xFF;
        packet[3] = (size)&0xFF;
        std::copy(data, data + size, packet.begin() + 4);

        // calculate the md5 checksum
        unsigned char md5sum[16];   // 16 bytes for MD5 checksum
        MD5(data, size, md5sum);  // Calculate MD5 checksum
        // Convert to hex string for printing
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < 16; ++i) {
            ss << std::setw(2) << static_cast<int>(md5sum[i]);
        }
        std::cout << "MD5: " << ss.str() << std::endl;

        printTimeMs("Before Send");
        sender_ptr->sendData(packet);
        printTimeMs("After Send");
        std::cout << "Sent " << size << " bytes of H.264 data" << std::endl;
      } catch (const TCPException &e) {
        printErrorAndQuit(e.what());
      } catch (const std::exception &e) {
        printErrorAndQuit("Unexpected error during sendData: " +
                          std::string(e.what()));
      }
    }

    gst_buffer_unmap(buffer, &map);
  }

  if (buffer) {
    GstClockTime timestamp = GST_BUFFER_PTS(buffer);
    std::cout << "Encoded frame at timestamp: "
              << GST_TIME_AS_MSECONDS(timestamp) << " ms" << std::endl;
  }

  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
  printTimeMs("Start");
  gst_init(&argc, &argv);
  signal(SIGINT, handle_sigint);

  // Check for --preview flag
  bool preview_enabled = false;
  std::string server_ip = "127.0.0.1";
  int server_port = 12345;

  bool listen_enabled = false;
  std::string listen_address = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--preview") {
      preview_enabled = true;
    } else if (arg == "--send") {
      send_enabled = true;
      encoding_enabled = true;
    } else if (arg == "--listen" && i + 1 < argc) {
      listen_enabled = true;
      listen_address = argv[++i];
    } else if (arg == "--server" && i + 1 < argc) {
      server_ip = argv[++i];
      // cache the ip
      send_to_server = std::string(server_ip);
    } else if (arg == "--port" && i + 1 < argc) {
      server_port = std::stoi(argv[++i]);
      // cache the port
      send_to_port = server_port;
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  --preview      Enable video preview\n";
      std::cout << "  --listen       Listen to control commands\n";
      std::cout << "  --send         Enable sending encoded video over TCP\n";
      std::cout << "  --server IP    Server IP address (default: 127.0.0.1)\n";
      std::cout << "  --port PORT    Server port (default: 12345)\n";
      std::cout << "  --help         Show this help message\n";
      return 0;
    }
  }

  // Correct logic: disable sending when listen is enabled
  // as the sending is controlled by the remote side (headset)
  if (listen_enabled) {
    send_enabled = false;
    encoding_enabled = false;
  }

  if (listen_enabled) {
    // Initialize TCPServer
    server_ptr = std::unique_ptr<TCPServer>(new TCPServer(listen_address));
    server_ptr->setDataCallback(onDataCallback);
    server_ptr->setDisconnectCallback(onDisconnectCallback);
    server_ptr->start();
    std::cout << "TCPServer is listening on " << listen_address << std::endl;
  }

  if (send_enabled && !initialize_sender()) {
    return -1;
  }

  // Initialize ZED
  sl::Camera zed;
  sl::InitParameters init_params;
  init_params.camera_resolution = sl::RESOLUTION::HD720;
  init_params.camera_fps = 60;
  if (zed.open(init_params) != sl::ERROR_CODE::SUCCESS) {
    std::cerr << "Failed to open ZED camera\n";
    return -1;
  }

  // Construct pipeline string based on preview flag
  // `insert-sps-pps=true idrinterval=15` are necessary for the headset decoding
  // insert-sps-pps=true: Makes sure every keyframe (IDR) is preceded by SPS/PPS
  // — necessary for streaming and decoding startup. idrinterval=15: Forces a
  // keyframe every 15 frames (adjust as needed).

  std::string pipeline_str =
      "appsrc name=mysource is-live=true format=time "
      "caps=video/x-raw,format=BGRA,width=2560,height=720,framerate=60/1 ! "
      "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
      "tee name=t "
      "t. ! queue ! nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true "
      "idrinterval=15 bitrate=4000000 ! h264parse ! appsink name=mysink "
      "emit-signals=true sync=false ";

  if (preview_enabled) {
    pipeline_str += "t. ! queue ! "
                    "nvvidconv ! videoconvert ! "
                    "autovideosink sync=false ";
    encoding_enabled = true;
  }

  // Launch pipeline
  GError *error = nullptr;
  GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
  if (!pipeline) {
    std::cerr << "Failed to create pipeline: " << error->message << std::endl;
    g_clear_error(&error);
    return -1;
  }

  // Bind appsrc/appsink
  GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
  GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");

  g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), nullptr);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  sl::Mat zed_image;
  int frame_id = 0;

  while (!stop_requested) {
    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
      auto grabbed = printTimeMs("Grabbed", false);
      zed.retrieveImage(zed_image, sl::VIEW::SIDE_BY_SIDE);
      cv::Mat cv_image =
          slMat2cvMat(zed_image); // ZED OpenCV conversion -> BGRA

      auto befored = printTimeMs("Before Encoding", false);
      if (encoding_enabled) {
        printf(grabbed.c_str());
        printf(befored.c_str());
        GstBuffer *buffer = gst_buffer_new_allocate(
            nullptr, cv_image.total() * cv_image.elemSize(), nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        memcpy(map.data, cv_image.data, cv_image.total() * cv_image.elemSize());
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) =
            gst_util_uint64_scale(frame_id, GST_SECOND, 60);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, 60);
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);

        frame_id++;
      }
    }
  }

  if (send_enabled && sender_ptr) {
    sender_ptr->disconnect();
    std::cout << "Disconnected from server" << std::endl;
  }

  // close the server if it was created
  if (listen_enabled && server_ptr) {
    server_ptr->stop();
    std::cout << "TCPServer stopped" << std::endl;
  }

  // Clean shutdown
  gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(appsrc);
  gst_object_unref(appsink);
  gst_object_unref(pipeline);
  zed.close();

  return 0;
}