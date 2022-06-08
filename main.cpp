#include <iostream>
#include <vector>
#include <Vec3.h>
#include <utils.h>


int main() {

    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    Vec3 black = {0, 0, 0};
    Vec3 white = {1, 1, 1};
    std::vector<Vec3> data(image_height * image_width, black);

    for (int row = 0; row < image_height; ++row) {
        for (int col = 0; col < image_width; ++col) {
            if ((image_width / 4 <= col && col <= image_width / 2) && (image_height / 4 <= row && row <= image_height / 2)) {
                data[row * image_width + col] = white;
            }
        }
    }

    std::string filename = "test.jpg";
    int channelCount = 3;
    saveImgAsJpg(filename, data, image_width, image_height, channelCount);
    std::cout << "Done." << "\n";
    return 0;
}
