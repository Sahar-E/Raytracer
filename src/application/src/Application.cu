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
#include <glm/gtx/quaternion.hpp>
#include "commonDefines.h"
#include "Vec3.cuh"
#include "World.cuh"
#include "Camera.cuh"
#include "RayTracerRenderer.cuh"
#include "InputHandler.h"


//void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
//{
//    if (action == GLFW_PRESS || action == GLFW_REPEAT)
//        Application::getInstance().handleKeyboardEvent(key);
//}




int Application::start(const Configurations &configurations) {
    _config = configurations;
    assert(0 < _config.rayBounces && _config.rayBounces <= MAX_BOUNCES);

    Vec3 vUp = {0, 1, 0};
    Vec3 lookFrom = {0, 0.5, 2.5};
    Vec3 lookAt = {0., 0.2, -2};
    float focusDist = (lookFrom - lookAt).length();


    auto world = World::initWorld1();
    std::cout << "Size: " << world.getTotalSizeInMemoryForObjects() << "\n";
    std::cout << "nSpheres: " << world.getNSpheres()  << "\n";
    assert(world.getTotalSizeInMemoryForObjects() < 48 * pow(2, 10) && "There is a hard limit for NVIDIA's shared memory size of 48KB for one block.");
    _camera = std::make_shared<Camera>(lookFrom, lookAt, vUp, _config.aspectRatio, _config.vFov, _config.aperture, focusDist);
    _rayTracerRenderer = std::make_shared<RayTracerRenderer>(_config.image_width, _config.image_height, world, _camera, _config.rayBounces);


//    GLFWwindow *_window;
    const char *glsl_version;
    if(getGLWindow(_window, _config.aspectRatio, glsl_version) == -1) {
        return -1;
    }
    InputHandler inputHandler(_window);

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
        LiveTexture texture(_rayTracerRenderer->getPixelsOutAsChars(), _config.image_width, _config.image_height, GL_RGB);
        texture.bind();
        shader.setUniform1i("u_texture", 0);

        va.unbind();
        vb.unbind();
        ib.unbind();
        shader.unbind();

        imguiInit(_window, glsl_version);
        GUIRenderer guiRenderer;

        glm::vec3 cameraLookFrom = glm::vec3(lookFrom.x(), lookFrom.y(), lookFrom.z());
        glm::vec3 cameraLookAt = glm::vec3(lookAt.x(), lookAt.y(), lookAt.z());

//        glfwSetKeyCallback(window, key_callback);

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(_window)) {
            startImguiFrame();

            /* Render here */
            guiRenderer.clear();
            {
//                TimeThis t("Update texture");
                _rayTracerRenderer->render();
                _rayTracerRenderer->syncPixelsOut();
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
                ImGui::SliderFloat3("Camera LookFrom (x,y,z) - disabled", &cameraLookFrom.x, -15.0f, 15.0f);
                ImGui::SliderFloat3("Camera LookAt (x,y,z) - disabled", &cameraLookAt.x, -15.0f, 15.0f);
                Vec3 sliderLookFrom = {cameraLookFrom.x, cameraLookFrom.y, cameraLookFrom.z};
                Vec3 sliderLookAt = {cameraLookAt.x, cameraLookAt.y, cameraLookAt.z};
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                            ImGui::GetIO().Framerate);
                bool cameraChanged = false;
//                std::cout << inputHandler.getMouseX() << ", " << inputHandler.getMouseY() << "\n";
                std::cout << ImGui::GetIO().MouseDelta.x << ", " << ImGui::GetIO().MouseDelta.y << "\n";
                if (inputHandler.isMouseMove() && !ImGui::GetIO().WantCaptureMouse) {
                    glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    float speedFactor = 0.08f;
                    float dX = ImGui::GetIO().MouseDelta.x * speedFactor;
                    float dY = ImGui::GetIO().MouseDelta.y * speedFactor;
                    if (dX != 0.0f || dY != 0.0f) {
                        _camera->rotateCamera(-dX, dY);
                        cameraChanged = true;
                    }
                } else {
                    int state = glfwGetInputMode(_window, GLFW_CURSOR);
                    if (state == GLFW_CURSOR_DISABLED) {
                        glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                    }
                }


                if (ImGui::Button("left")) {
                    _camera->rotateCamera(-1.f, 0.0f);
                    cameraChanged = true;
                }
                if (ImGui::Button("right")) {
                    _camera->rotateCamera(1.f, 0.0f);
                    cameraChanged = true;
                }
                if (ImGui::Button("up")) {
                    _camera->rotateCamera(0.0f, -1.f);
                    cameraChanged = true;
                }
                if (ImGui::Button("down")) {
                    _camera->rotateCamera(0.0f, 1.f);
                    cameraChanged = true;
                }

                if (inputHandler.isKeyDown(GLFW_KEY_A)) {
                    _camera->moveCameraRight(-.1f);
                    cameraChanged = true;
                }
                if (inputHandler.isKeyDown(GLFW_KEY_D)) {
                    _camera->moveCameraRight(.1f);
                    cameraChanged = true;
                }
                if (inputHandler.isKeyDown(GLFW_KEY_W)) {
                    _camera->moveCameraForward(.1f);
                    cameraChanged = true;
                }
                if (inputHandler.isKeyDown(GLFW_KEY_S)) {
                    _camera->moveCameraForward(-.1f);
                    cameraChanged = true;
                }
                if (inputHandler.isKeyDown(GLFW_KEY_SPACE) && !inputHandler.isKeyDown(GLFW_KEY_LEFT_SHIFT)) {
                    _camera->moveCameraUp(.1f);
                    cameraChanged = true;
                }
                if (inputHandler.isKeyDown(GLFW_KEY_SPACE) && inputHandler.isKeyDown(GLFW_KEY_LEFT_SHIFT)) {
                    _camera->moveCameraUp(-.1f);
                    cameraChanged = true;
                }


                if (cameraChanged) {
                    _rayTracerRenderer->clearPixels();
                }

                ImGui::End();
            }


            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


            /* Swap front and back buffers */
            glfwSwapBuffers(_window);
            /* Poll for and process events */
            glfwPollEvents();
        }
        imguiCleanup();
    }


    glfwTerminate();
    return 0;
}


Application &Application::getInstance() {
    static Application INSTANCE;
    return INSTANCE;
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

const std::shared_ptr<Camera> &Application::getCamera() const {
    return _camera;
}

const std::shared_ptr<RayTracerRenderer> &Application::getRayTracerRenderer() const {
    return _rayTracerRenderer;
}

GLFWwindow *Application::getWindow() const {
    return _window;
}
