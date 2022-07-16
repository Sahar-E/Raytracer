// TODO-Sahar: Testing OpenGL...
#include <vector>
#include <World.cuh>
#include <Camera.cuh>
#include <Renderer.cuh>
#include "utils.h"
#include "TimeThis.h"
#include "commonDefines.h"
#include <string>
#include <cassert>
//#include "cuda_runtime_api.h"
//#include "commonCuda.cuh"
//
//
//
//#define GLEW_BUILD
//#include <gl/GL.h>
//#include <gl/GL.h>


#include <iostream>
#include <imgui-docking/include/imgui.h>
#include <imgui-docking/include/imgui_impl_glfw.h>
#include <imgui-docking/include/imgui_impl_opengl3.h>
#include "glew-2.1.0/include/GL/glew.h"
#include "glfw-3.3.7/include/GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <regex>
#include "commonOpenGL.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"
#include "GUIRenderer.h"
#include "Texture.h"
#include "stb_library/include/stb_library/stb_image.h"
#include "LiveTexture.h"


void copyRGBToCharArray(std::shared_ptr<unsigned char[]> &pixelsOutAsChars, const Color *pixelsOut, int nPixels) {
    int channelCount = 3;
    for (int i = 0; i < nPixels; ++i) {
        Color pixel = pixelsOut[i];
        pixel = clamp(gammaCorrection(pixel), 0.0, 0.999);
        pixelsOutAsChars[i * channelCount] =     static_cast<unsigned char>(pixel.x() * 255);
        pixelsOutAsChars[i * channelCount + 1] = static_cast<unsigned char>(pixel.y() * 255);
        pixelsOutAsChars[i * channelCount + 2] = static_cast<unsigned char>(pixel.z() * 255);
    }
}

int main() {


    TimeThis timeThis;
    const auto aspectRatio = 3.0f / 2.0f;
    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / aspectRatio);
    const int rayBounces = 7;
    float vFov = 26.0f;
    float aperture = 0.005f;
    int nFrames = 1;

    assert(0 < rayBounces && rayBounces <= MAX_BOUNCES);

    Vec3 vUp = {0, 1, 0};
    Vec3 lookFrom = {0, 0.5, 2.5};
    Vec3 lookAt = {0., 0.2, 0};
    float focusDist = (lookFrom - lookAt).length();



    auto world = World::initWorld1();
    std::cout << "Size: " << world.getTotalSizeInMemoryForObjects() << "\n";
    std::cout << "nSpheres: " << world.getNSpheres()  << "\n";
    assert(world.getTotalSizeInMemoryForObjects() < 48 * pow(2, 10) && "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");
    auto camera = Camera(lookFrom, lookAt, vUp, aspectRatio, vFov, aperture, focusDist);
    Renderer renderer(image_width, image_height, world, camera, rayBounces);

//    for (int j = 0; j < nFrames; ++j) {
//        renderer.render();
//        std::cout << "Done iteration #: " << j  << "\n";
//    }




//    std::cout << "Hello" << std::endl;
    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit()) { return -1; }

    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);


    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(1920 , static_cast<int>(1920.0f / aspectRatio), "RayTracer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK) {
        std::cerr << "glewInit() failed\n";
    }


    std::cout <<  "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    {
        float positions[] = {
                -0.9999f, -0.9999f, 0.0f, 0.0f,
                0.9999f, -0.9999f, 1.0f, 0.0f,
                0.9999f, 0.9999f, 1.0f, 1.0f,
                -0.9999f, 0.9999f, 0.0f, 1.0f
        };
        unsigned int indices[] = {
                0, 1, 2,
                2, 3, 0
        };

        checkGLErrors(glEnable(GL_BLEND));
        checkGLErrors(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

        VertexArray va;
        VertexBuffer vb(positions, sizeof(float) * 4 * 4);

        VertexBufferLayout layout;
        layout.push<float>(2);
        layout.push<float>(2);
        va.addBuffer(vb, layout);

        IndexBuffer ib(indices, 6);

        glm::mat4 proj = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(-0,0,0));

        Shader shader("resources/shaders/Basic.shader");
        shader.bind();
//        shader.setUniform4f("u_color", 0.2f, 0.2f, 0.9f, 1.0f);


        stbi_set_flip_vertically_on_load(true);
        int w, h, channelsInFile;
        auto p = std::shared_ptr<unsigned char[]>(stbi_load("resources/textures/img.png", &w, &h, &channelsInFile, 4));

        int nPixels = renderer.getNPixelsOut();
        auto rgbAsChars = std::shared_ptr<unsigned char[]>(new unsigned char[nPixels * 3]);

        renderer.render();
        copyRGBToCharArray(rgbAsChars, renderer.getPixelsOut(), nPixels);
        LiveTexture texture(rgbAsChars, image_width, image_height, GL_RGB);

//        LiveTexture texture(p, w, h, GL_RGBA);

//        Texture texture("resources/textures/img.png");
        texture.bind();
        shader.setUniform1i("u_texture", 0);

        va.unbind();
        vb.unbind();
        ib.unbind();
        shader.unbind();

        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
        ImGui::StyleColorsDark();

        // Our state
        bool show_demo_window = true;
        bool show_another_window = false;
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        GUIRenderer guiRenderer;

        glm::vec3 translation = glm::vec3(0, 0, 0);
        float r = 0.0f;
        float increment = 0.05f;

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window)) {
            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            /* Render here */
            guiRenderer.clear();

//            texture.bind();
            renderer.render();
            copyRGBToCharArray(rgbAsChars, renderer.getPixelsOut(), nPixels);
            texture.updateTexture();
//            LiveTexture texture(rgbAsChars, image_width, image_height);

            glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
            model = glm::scale(model, glm::vec3(1.0f, -1.0f, 1.0f));
            glm::mat4 mvp = proj * view * model;

            shader.bind();
//            shader.setUniform4f("u_color", r, 0.2f, 0.2f, 1.0f);
            shader.setUniformMat4f("u_mvpMatrix", mvp);

            guiRenderer.draw(va, ib, shader);

            if (r > 1.0f || r < 0.0f) {
                increment = -increment;
            }
            r += increment;

            // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
            {
                static float f = 0.0f;
                static int counter = 0;

                ImGui::Begin("SmallWindow!");                          // Create a window called "Hello, world!" and append into it.

//                ImGui::Text(
//                        "This is some useful text.");               // Display some text (you can use a format strings too)
//                ImGui::Checkbox("Demo Window",
//                                &show_demo_window);      // Edit bools storing our window open/close state
//                ImGui::Checkbox("Another Window", &show_another_window);

//                ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
                ImGui::SliderFloat3("Translation (x,y,z)", &translation.x, -2.0f, 2.0f);
//                ImGui::ColorEdit3("clear color", (float *) &clear_color); // Edit 3 floats representing a color

//                if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
//                {
//                    counter++;
//                }
//                ImGui::SameLine();
//                ImGui::Text("counter = %d", counter);
//
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                            ImGui::GetIO().Framerate);
                ImGui::End();
            }


            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
        }
        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }


    glfwTerminate();
    return 0;
}


//int main() {
//    TimeThis timeThis;
//    const auto aspectRatio = 3.0f / 2.0f;
//    const int image_width = 1200;
//    const int image_height = static_cast<int>(image_width / aspectRatio);
//    const int rayBounces = 7;
//    float vFov = 26.0f;
//    float aperture = 0.005f;
//    int nFrames = 1;
//
//    assert(0 < rayBounces && rayBounces <= MAX_BOUNCES);
//
//    Vec3 vUp = {0, 1, 0};
//    Vec3 lookFrom = {0, 0.5, 2.5};
//    Vec3 lookAt = {0., 0.2, 0};
//    float focusDist = (lookFrom - lookAt).length();
//
//
//
//    auto world = World::initWorld1();
//    std::cout << "Size: " << world.getTotalSizeInMemoryForObjects() << "\n";
//    std::cout << "nSpheres: " << world.getNSpheres()  << "\n";
//    assert(world.getTotalSizeInMemoryForObjects() < 48 * pow(2, 10) && "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");
//    auto camera = Camera(lookFrom, lookAt, vUp, aspectRatio, vFov, aperture, focusDist);
//    Renderer renderer(image_width, image_height, world, camera, rayBounces);
//
//    for (int j = 0; j < nFrames; ++j) {
//        renderer.render();
//        std::cout << "Done iteration #: " << j  << "\n";
//    }
//
//    std::string filename = "test.jpg";
//    int channelCount = 3;
//    std::vector<std::tuple<float, float, float>> rgb(renderer.getNPixelsOut(), {0, 0, 0});
//    for (int i = 0; i < renderer.getNPixelsOut(); ++i) {
//        Color pixel = renderer.getPixelsOut()[i];
//        pixel = clamp(gammaCorrection(pixel), 0.0, 0.999);
//        rgb[i] = {pixel.x(), pixel.y(), pixel.z()};
//    }
//    saveImgAsJpg(filename, rgb, image_width, image_height, channelCount);
//
//    std::cout << "Done." << "\n";
//    return 0;
//}