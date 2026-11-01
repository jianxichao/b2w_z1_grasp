#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <unitree/idl/ros2/String_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <network-interface>\n";
    return 2;
  }

  const std::string interface_name = argv[1];
  constexpr const char *topic_name = "rt/aruco_cmd";

  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, interface_name);
    unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_> publisher(
        topic_name);
    publisher.InitChannel();

    // Give DDS discovery time to match the already-running recharge reader.
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std_msgs::msg::dds_::String_ message;
    message.data("aruco_start");
    const bool written = publisher.Write(message, 1000000);

    std::cout << "topic=" << topic_name << " interface=" << interface_name
              << " payload=aruco_start write_return="
              << (written ? "true" : "false") << std::endl;

    // Keep the writer alive briefly so a best-effort sample can leave the
    // process before DDS entities are destroyed.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    publisher.CloseChannel();
    unitree::robot::ChannelFactory::Instance()->Release();
    return written ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "native DDS publisher failed: " << error.what() << std::endl;
    return 1;
  }
}
