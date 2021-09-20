--------------------------------------------------------------------
# CS380 GPU and GPGPU Programming
### Programming Assignment #3
### Deferred Shading and Image Processing with GLSL
### Contacts: peter.rautek@kaust.edu.sa
--------------------------------------------------------------------

# Introduction:

In this assignment you will implement variations of a rendering technique called Deferred Shading.
The idea of Deferred Shading is to split the computations into two rendering passes, one geometry pass and one shading pass.

# Single-pass vs. Multi-pass rendering

Computing partial results of an output image using the OpenGL graphics pipeline is often called a rendering pass.
Single-pass rendering needs only one pass to compute the output image which is directly displayed on the screen.
Multi-pass rendering computes the output image using multiple render-passes. Each pass computes a partial result of the output image using the result of the previous passes. Intermediate results are stored in frame buffer objects (FBOs) that are not displayed on the screen.

# Deferred Shading

Deferred Shading is a versatile two-pass rendering technique. The first pass renders the geometry to offscreen buffers. These buffers are not visible to the user.
We store information like depth, color, normal, etc. for each fragment in these buffers.
We use these buffers in the second render-pass to compute the image that will be displayed on the screen.
Because this second render-pass is typically only computed once per fragment, we can apply more compute intensive shading techniques such as: Screen Space Ambient Occlusion (SSAO), more advanced shading models with many light sources, image processing operations, camera effects such as Depth of Field, etc.


# Tasks:

1. Start by setting up a simple shader in order to see the preloaded scene on the screen (you should see 21 balloons).

2. Setup your framebuffers and implement the geometry rendering-pass [1].
You will need buffers for colors, normals, and depth. The depth should be normalized (see [3] on how depth values are computed).

3. Setup the geometry (i.e., a rectangle) that fills the entire screen and render it in the second pass [1].

4. Implement at least four variations of the second render-pass [1].
4.a) Compute phong shading in the second render-pass (you need to store light parameters as uniforms, and use the color and the normal from the buffers).
4.b) Implement edge detection on the depth buffer and display the result (e.g., Chapter 19.7. [5]).
4.c) Implement blurring using a variable sized box (or mean) filter (e.g., Chapter 19.7. [5]).
4.d) Implement the Depth of Field effect ([4] "Reverse-mapped z-buffer techniques").
The idea of the Depth of Field technique is to imitate the blurring of objects that are out of focus when using real cameras.
You can implement it by defining a focus plane (uniform variable in the range [0..1]). Compute the distance between the focus plane and the normalized depth value in the depth buffer (computed in the first pass). Use the blur filter implemented in 4.c) with a variable kernel size. Map zero distance to a kernel of size 1x1, and the highest distances to a kernel of size 15x15.

5. Provide key mappings to allow the user to switch between modes and to change the focal-plane for 4.d).

6. Submit your program and a report including screenshots of the results.


# Notes:

- You don't need to read the referenced papers in the GPU Gems website.
- The final result might have artifacts on the borders of the image (this can be fixed but is fine for your implementation).

# BONUS:

- Implement Screen Space Ambient Occlusion [1]

# References:
[1] https://learnopengl.com/Advanced-Lighting/Deferred-Shading

[2] https://medium.com/swlh/how-image-blurring-works-652051aee2d1

[3] https://learnopengl.com/Advanced-OpenGL/Depth-testing

[4] https://developer.nvidia.com/gpugems/gpugems/part-iv-image-processing/chapter-23-depth-field-survey-techniques

[5] OpenGL Shading Language, by Rost, Randi J; Licea-Kane, Bill, 2010:
https://learning.oreilly.com/library/view/opengl-shading-language/9780321669247/ch19.html