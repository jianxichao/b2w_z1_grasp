// Camera service requests only. Never creates a SportClient or motion request.
#include <unitree/robot/b2/front_video/front_video_client.hpp>
#include <unitree/robot/b2/back_video/back_video_client.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

template <class Client>
bool Capture(Client& client, const std::filesystem::path& path) {
    client.SetTimeout(3.0f);
    client.Init();
    std::vector<uint8_t> data;
    const int code = client.GetImageSample(data);
    if (code != 0 || data.empty()) {
        std::cerr << path.filename() << ": camera response code=" << code << " bytes=" << data.size() << '\n';
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
    if (!out.good()) return false;
    std::cout << "Saved " << path << " (" << data.size() << " bytes)\n";
    return true;
}
int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: capture_builtin NETWORK_INTERFACE OUTPUT_DIRECTORY [front|back|both]\n";
        return 2;
    }
    const std::string camera = argc == 4 ? argv[3] : "both";
    if (camera != "front" && camera != "back" && camera != "both") return 2;
    const std::filesystem::path output(argv[2]);
    std::filesystem::create_directories(output);
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    bool ok = true;
    if (camera == "front" || camera == "both") {
        unitree::robot::b2::FrontVideoClient client;
        ok = Capture(client, output / "front.jpg") && ok;
    }
    if (camera == "back" || camera == "both") {
        unitree::robot::b2::BackVideoClient client;
        ok = Capture(client, output / "back.jpg") && ok;
    }
    return ok ? 0 : 1;
}
