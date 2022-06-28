#include <iostream>
#include <vector>
#include <World.cuh>
#include <Camera.cuh>
#include <Renderer.cuh>
#include "utils.cuh"
#include <chrono>
#include <c++/9/algorithm>

//World initWorld1() {
////    auto world = World();
//    Color red = {0.8, 0.2, 0.1};
//    Color white = {1, 1, 1};
//    Color gold1 = {212 / 255.0, 175 / 255.0, 55 / 255.0};
//    Color gold2 = {218 / 255.0, 165 / 255.0, 32 / 255.0};
//    Color bezh = {227 / 255.0, 203 / 255.0, 165 / 255.0};
//    Color green = {51 / 255.0, 83 / 255.0, 69 / 255.0};
//    Color brown = {118 / 255.0, 91 / 255.0, 70 / 255.0};
//    Color redBrown = {141 / 255.0, 78 / 255.0, 44 / 255.0};
//    Color darkGreen = {74 / 255.0, 71 / 255.0, 51 / 255.0};
//    Color neonPurple = {176 / 255.0, 38 / 255.0, 255 / 255.0};
//    Color neonGreen = {57 / 255.0, 255 / 255.0, 20 / 255.0};
//
////    Material metalGreen = Material(whiteGreenish, 0.99);
//    Material mirror = Material::getSpecular(white, white, 0.0, 1.0);
//    Material metalRedBrown = Material::getSpecular(redBrown, redBrown, 0.3, 0.3);
//    Material metalGold = Material::getSpecular(gold1, gold2, 0.25, 1);
//    Material lambertianBezh = Material::getLambertian(bezh);
//    Material lambertianGreen = Material::getLambertian(green);
//    Material lambertianBrown = Material::getLambertian(brown);
//    Material lambertianRedBrown = Material::getLambertian(redBrown);
//    Material metalGreen = Material::getSpecular(green, green, 0.2, 0.3);
//    Material lambertianDarkGreen = Material::getLambertian(darkGreen);
//
//    Material neonPurpleGlow = Material::getGlowing(white * .8, neonPurple, 4);
//    Material neonGreenGlow = Material::getGlowing(white * .8, neonGreen, 4);
//    Material whiteLight = Material::getGlowing(white * .8, white, 10);
//
//    Material glass = Material::getGlass(white, 1.5);
//
//    world.push_back(Sphere({4, 4.5, -3}, 0.5, whiteLight));
//
//    world.push_back(Sphere({0, 0, -2}, 0.5, lambertianRedBrown));
//    world.push_back(Sphere({1.2, 0, -3}, 0.5, mirror));
//    world.push_back(Sphere({-1.2, -0.2, -2.5}, 0.3, metalGold));
//
//    world.push_back(Sphere({0.6, -0.4, -2.2}, 0.1, lambertianDarkGreen));
//    world.push_back(Sphere({0.4, -0.4, -1.5}, 0.1, lambertianBezh));
//    world.push_back(Sphere({0.5, -0.4, -1.2}, 0.1, glass));
//    world.push_back(Sphere({1, -0.4, -1.5}, 0.1, metalGreen));
//    world.push_back(Sphere({-0.9, -0.4, -3.3}, 0.1, lambertianGreen));
//    world.push_back(Sphere({-0.5, -0.4, -1.8}, 0.1, mirror));
//    world.push_back(Sphere({-0.8, -0.4, -1.7}, 0.1, neonPurpleGlow));
//    world.push_back(Sphere({-0.15, -0.4, -1.1}, 0.1, lambertianGreen));
//    world.push_back(Sphere({-0.45, -0.4, -1.2}, 0.1, glass));
//    world.push_back(Sphere({0.15, -0.4, -1.45}, 0.1, neonGreenGlow));
//    world.push_back(Sphere({-0.8, -0.4, -1.3}, 0.1, lambertianRedBrown));
//    world.push_back(Sphere({0, -1000.5, -2}, 1000, lambertianBrown));
//    return world;
//}


World initWorld2() {
    std::vector<Sphere> sphereList{};
    Color brown = {118 / 255.0, 91 / 255.0, 70 / 255.0};
    Material lambertianBrown = Material::getLambertian(brown);
    auto floor = Sphere({0, -5000, 0}, 5000, lambertianBrown);
    sphereList.push_back(floor);


    Color white = {1, 1, 1};
    Material whiteLight = Material::getGlowing(white * .8, white, 10);
    sphereList.push_back(Sphere({4, 5, 4}, 1, whiteLight));


    double radius = 1;
    std::vector<Vec3> bigBallsLocs = {{1.5,  radius, 4},
                                      {0.5,  radius, 2},
                                      {-0.5, radius, 0}};
    Material mirror = Material::getSpecular(white, white, 0.0, 1.0);
    sphereList.emplace_back(Sphere(bigBallsLocs[0], radius, mirror));
    Color redBrown = {141 / 255.0, 78 / 255.0, 44 / 255.0};
    Material lambertianRedBrown = Material::getLambertian(redBrown);
    sphereList.emplace_back(Sphere(bigBallsLocs[1], radius, lambertianRedBrown));
    Material glass = Material::getGlass(white, 1.5);
    sphereList.emplace_back(Sphere(bigBallsLocs[2], radius, glass));

    double smallSphereRadius = 0.2;
    int randState = 1;
    for (int xLoc = -8; xLoc < 8; ++xLoc) {
        for (int zLoc = -17; zLoc < 12; ++zLoc) {
            bool locIsFree = !std::any_of(bigBallsLocs.begin(), bigBallsLocs.end(), [xLoc, zLoc](const Vec3 &ballLoc) {
                return fabs(ballLoc[0] - xLoc) < 1 && fabs(ballLoc[2] - zLoc) < 1;
            });
            if (locIsFree) {
                Point3 sphereLoc = {xLoc + 0.7 * randomDouble(randState), smallSphereRadius, zLoc + 0.7 * randomDouble(
                        randState)};
                double randomMaterialChooser = randomDouble(randState);
                Material mat;
                if (randomMaterialChooser < 0.5) {
                    auto albedo = randomVec(0.0, 1.0, randState) * randomVec(0.0, 1.0, randState);
                    mat = Material::getLambertian(albedo);
                } else if (randomMaterialChooser < 0.8) {
                    auto albedo = randomVec(0.0, 1.0, randState) * randomVec(0.0, 1.0, randState);
                    auto specularColor = albedo + randomVec(0.0, 1.0, randState) * 0.2;
                    mat = Material::getSpecular(albedo, specularColor, randomDouble(randState), randomDouble(randState));
                } else if (randomMaterialChooser < 0.9) {
                    auto albedo = randomVec(0.0, 1.0, randState) * randomVec(0.0, 1.0, randState);
                    auto emittedColor = randomVec(0.0, 1.0, randState) * randomVec(0.0, 1.0, randState);
                    mat = Material::getGlowing(albedo, emittedColor, 8);
                } else {
                    mat = Material::getGlass(white, 1.5);
                }
                sphereList.emplace_back(Sphere(sphereLoc, smallSphereRadius, mat));
            }
        }
    }

    return {sphereList.data(), sphereList.size()};
}

int main() {
    auto start = std::chrono::steady_clock::now();
    const auto aspectRatio = 3.0 / 2.0;
    const int image_width = 300;
    const int image_height = static_cast<int>(image_width / aspectRatio);
    const int rayBounces = 7;
    int nSamplesPerPixel = 100;
    int vFov = 26;
    Vec3 vUp = {0, 1, 0};

    Vec3 lookFrom = {0, 1.8, 12};
    Vec3 lookAt = {0., 0, 0};
    double aperture = 0.05;
    double focusDist = 10.0;

    auto world = initWorld2();
    auto camera = Camera(lookFrom, lookAt, vUp, aspectRatio, vFov, aperture, focusDist);
    Renderer renderer(image_width, image_height, world, camera, rayBounces, nSamplesPerPixel);
    std::vector<Color> renderedImage = renderer.render();

    std::string filename = "test.jpg";
    int channelCount = 3;
    std::vector<std::tuple<double, double, double>> rgb(renderedImage.size());
    for (int i = 0; i < renderedImage.size(); ++i) {
        rgb[i] = {renderedImage[i].x(), renderedImage[i].y(), renderedImage[i].z()};
    }

    saveImgAsJpg(filename, rgb, image_width, image_height, channelCount);

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Done." << "\n";
    std::cout << "Duration (ms): " << duration << "\n";
    return 0;
}



//// function to add the elements of two arrays
//__global__
//void add(int n, float *x, float *y) {
//    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
//         i < n;
//         i += blockDim.x * gridDim.x) {
////    for (int i = 0; i < n ; ++i) {
//        y[i] = x[i] + y[i];
//    }
//}


//int main() {
//    // Testing.
//    int N = 1 << 23; // 1M elements
//
////    float *x = new float[N];
////    float *y = new float[N];
//    float *x, *y;
//    cudaMallocManaged(reinterpret_cast<void **>(&x), N * sizeof(float));
//    cudaMallocManaged(reinterpret_cast<void **>(&y), N * sizeof(float));
//
//
//    // initialize x and y arrays on the host
//    for (int i = 0; i < N; i++) {
//        x[i] = 1.0f;
//        y[i] = 2.0f;
//    }
//
//    // Run kernel on 1M elements on the CPU
//    auto start = std::chrono::steady_clock::now();
////    add(N, x, y);
//    int blockSize = 256;
//    int numBlocks = (N + blockSize - 1) / blockSize;
//
//    add<<<numBlocks, blockSize>>>(N, x, y);
//    cudaDeviceSynchronize();
//
//    auto end = std::chrono::steady_clock::now();
//
//    // Check for errors (all values should be 3.0f)
//    float maxError = 0.0f;
//    for (int i = 0; i < N; i++)
//        maxError = fmax(maxError, fabs(y[i] - 3.0f));
//    std::cout << "Max error: " << maxError << "\n";
//
////    // Free memory
////    delete [] x;
////    delete [] y;
////    // Free memory
//    cudaFree(x);
//    cudaFree(y);
//
//    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//
//    std::cout << "Done." << "\n";
//    std::cout << "Duration (micro-s): " << duration << "\n";
//    return 0;
//}
