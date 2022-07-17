//
// Created by Sahar on 17/07/2022.
//

#include "Application.cuh"
#include <string>
#include <memory>
#include <imgui-docking/include/imgui.h>
#include <imgui-docking/include/imgui_impl_glfw.h>
#include <imgui-docking/include/imgui_impl_opengl3.h>
#include "glew-2.1.0/include/GL/glew.h"
#include "glfw-3.3.7/include/GLFW/glfw3.h"
#include "GUIRenderer.h"
#include <iostream>
#include <utility>
#include "commonOpenGL.h"
#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"
#include "stb_library/include/stb_library/stb_image.h"
#include "LiveTexture.h"
#include "../../ray_tracing_library/include/TimeThis.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "commonDefines.h"
#include "Vec3.cuh"
#include "World.cuh"
#include "Camera.cuh"
#include "RayTracerRenderer.cuh"

Application::Application() {}


int Application::start() {
    const auto aspectRatio = 3.0f / 2.0f;
    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / aspectRatio);
    const int rayBounces = 7;
    float vFov = 26.0f;
    float aperture = 0.005f;


    assert(0 < rayBounces && rayBounces <= MAX_BOUNCES);

    Vec3 vUp = {0, 1, 0};
    Vec3 lookFrom = {0, 0.5, 2.5};
    Vec3 lookAt = {0., 0.2, -2};
    float focusDist = (lookFrom - lookAt).length();



    auto world = World::initWorld1();
    std::cout << "Size: " << world.getTotalSizeInMemoryForObjects() << "\n";
    std::cout << "nSpheres: " << world.getNSpheres()  << "\n";
    assert(world.getTotalSizeInMemoryForObjects() < 48 * pow(2, 10) && "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");
    auto camera = Camera(lookFrom, lookAt, vUp, aspectRatio, vFov, aperture, focusDist);
    RayTracerRenderer rayTracerRenderer(image_width, image_height, world, camera, rayBounces);


    GLFWwindow *window;
    const char *glsl_version;
    if(getGLWindow(window, aspectRatio, glsl_version) == -1) {
        return -1;
    }
    initGlBlendingConfigurations();

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


        glm::mat4 proj = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(-0,0,0));

        VertexArray va;
        VertexBuffer vb(positions, sizeof(float) * 4 * 4);

        VertexBufferLayout layout;
        layout.push<float>(2);
        layout.push<float>(2);
        va.addBuffer(vb, layout);

        IndexBuffer ib(indices, 6);


        Shader shader("resources/shaders/Basic.shader");
        shader.bind();
        LiveTexture texture(rayTracerRenderer.getPixelsOutAsChars(), image_width, image_height, GL_RGB);
        texture.bind();
        shader.setUniform1i("u_texture", 0);

        va.unbind();
        vb.unbind();
        ib.unbind();
        shader.unbind();

        imguiInit(window, glsl_version);
        GUIRenderer guiRenderer;

        glm::vec3 cameraLookFrom = glm::vec3(lookFrom.x(), lookFrom.y(), lookFrom.z());
        glm::vec3 cameraLookAt = glm::vec3(lookAt.x(), lookAt.y(), lookAt.z());

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window)) {
            startImguiFrame();

            /* Render here */
            guiRenderer.clear();
            {
                TimeThis t("Update texture");
                rayTracerRenderer.render();
                rayTracerRenderer.syncPixelsOut();
                texture.updateTexture();
            }

            glm::mat4 model = glm::mat4(1.0f);
            model = glm::scale(model, glm::vec3(1.0f, -1.0f, 1.0f));
            glm::mat4 mvp = proj * view * model;
            shader.bind();
            shader.setUniformMat4f("u_mvpMatrix", mvp);
            guiRenderer.draw(va, ib, shader);


            // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
            {
                ImGui::Begin("SmallWindow!");
                ImGui::SliderFloat3("Camera LookFrom (x,y,z)", &cameraLookFrom.x, -15.0f, 15.0f);
                ImGui::SliderFloat3("Camera LookAt (x,y,z)", &cameraLookAt.x, -15.0f, 15.0f);
                Vec3 sliderLookFrom = {cameraLookFrom.x, cameraLookFrom.y, cameraLookFrom.z};
                Vec3 sliderLookAt = {cameraLookAt.x, cameraLookAt.y, cameraLookAt.z};
                if (!isZeroVec((sliderLookFrom - lookFrom)) || !isZeroVec((sliderLookAt - lookAt))) {
                    lookFrom = sliderLookFrom;
                    lookAt = sliderLookAt;
                    rayTracerRenderer.clearPixels();
                    rayTracerRenderer.setCamera(Camera(lookFrom, lookAt, vUp, aspectRatio, vFov, aperture, focusDist));
                }
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
        imguiCleanup();
    }


    glfwTerminate();
    return 0;
}

void Application::initGlBlendingConfigurations() const {
    checkGLErrors(glEnable(GL_BLEND));
    checkGLErrors(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
}

void Application::startImguiFrame() const {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Application::imguiInit(GLFWwindow *window, const char *glsl_version) const {
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImGui::StyleColorsDark();
}

void Application::imguiCleanup() const {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

int Application::getGLWindow(GLFWwindow *&window, const float aspectRatio, const char *&glsl_version) const {
    glsl_version = "#version 330";/* Initialize the library */
    if (!glfwInit()) { return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);


    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(1920, static_cast<int>(1920.0f / aspectRatio), "RayTracer", nullptr, nullptr);
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


    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    return 0;
}
