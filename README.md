# Ray Tracer
## Introduction
Still under construction :)

<br />
<div align="center">
<h3 align="center">Real Time Ray Tracer</h3>
  <p align="center">
    This Ray Tracer Renderer provides the ability to move freely in the scene with the keyboard and mouse, while 
    streaming live the results from the GPU to the screen. 
    <br />
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#Demo Videos and Screenshots">Demo Videos and Screenshots</a></li>
    <li><a href="#citations">Citations</a></li>
  </ol>
</details>


## About The Project

This project is a ray tracer / path tracer visualizer that provides a way to move freely in the scene with the keyboard and mouse
similar to a free camera in a world. The tracing algorithm is calculated on the GPU using the 
[CUDA C++ toolkit](https://developer.nvidia.com/cuda-toolkit). The program is written mainly in C++20 and CUDA-C++17 with cmake.
To present the result in realtime, I used [OpenGL](https://en.wikipedia.org/wiki/OpenGL) 
with [glfw](https://github.com/glfw/glfw), 
[glew](https://github.com/nigels-com/glew) to have a windowed program and also
[ImGui](https://github.com/ocornut/imgui) to have a simple UI.



<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
### Prerequisites
1. Make sure you have the [CUDA toolkit](https://developer.nvidia.com/how-to-cuda-c-cpp) installed and have NVIDIA nvcc 
compiler installed.
2. Have cmake 3.8+ installed.

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Sahar-E/Raytracer.git raytracer
   ```
2. cd raytracer && mkdir build && cd build
3. cmake ..
4. cmake --build .

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

Launch the application executable.

<p align="right">(<a href="#top">back to top</a>)</p>

## Demo Videos and Screenshots

Example video for Real Time raytracing:

<ADD_VIDEO>

Last rendered image:

![Image Render Example](test.jpg?raw=true "Image Render Example.")

Screenshot of the program:

![Screenshot Example](screenshot.png?raw=true "Screenshot Example")


<p align="right">(<a href="#top">back to top</a>)</p>

## Citations and sources

- [_Scratch Pixel_](https://www.scratchapixel.com/index.php?redirect)
- [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [NVIDIA CUDA Guides](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [_TheCherno - Youtube channel_](https://www.youtube.com/c/TheChernoProject)
- [_demofox blog about Programming, Graphics, Gamedev..._](https://blog.demofox.org/)

<p align="right">(<a href="#top">back to top</a>)</p>