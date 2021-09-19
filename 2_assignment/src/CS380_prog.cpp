// CS 380 - GPGPU Programming, KAUST
//
// Programming Assignment #2

// includes
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include <assert.h>

#include <math.h>

#include "glad/glad.h" 
#include "GLFW/glfw3.h" 
// includes glm
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/ext.hpp>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// framework includes
#include "vbocube.h"
#include "vbomesh.h"

// include glslprogram
#include "glslprogram.h"

// window size
const unsigned int gWindowWidth = 512;
const unsigned int gWindowHeight = 512;

// cursor position
float cursor_x;
float cursor_y;

//object scale factor
int objsf=1;
int radians=0;

//shader option
int shaderOption;

// a simple cube
VBOCube *m_pCube;

// a more complex mesh
VBOMesh *m_pMesh;

// glsl program 
GLSLProgram *prog;

struct viewaxis{
	float x;
	float y;
	float z;
};

viewaxis va={0,0,1};

// glfw error callback
void glfwErrorCallback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

// OpenGL error debugging callback
void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar *message,
	const void *userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}




// query GPU functionality we need for OpenGL, return false when not available
bool queryGPUCapabilitiesOpenGL()
{

	return true;
}

// query GPU functionality we need for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{

	return true;
}



// init application 
// - load application specific data 
// - set application specific parameters
// - initialize stuff
bool initApplication(int argc, char **argv)
{
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(glDebugOutput, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	
	
	std::string version((const char *)glGetString(GL_VERSION));
	std::stringstream stream(version);
	unsigned major, minor;
	char dot;

	stream >> major >> dot >> minor;
	
	assert(dot == '.');
	if (major > 3 || (major == 2 && minor >= 0)) {
		std::cout << "OpenGL Version " << major << "." << minor << std::endl;
	} else {
		std::cout << "The minimum required OpenGL version is not supported on this machine. Supported is only " << major << "." << minor << std::endl;
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, gWindowWidth, gWindowHeight);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
		
	return true;
}

// one time initialization to setup the 3D scene
void setupScene()
{
	// TODO: Set up a camera. Hint: The glm library is your friend
	 float radius = 10.0f;
	float camX = sin(glfwGetTime()) * radius;
	float camZ = cos(glfwGetTime()) * radius;
	glm::mat4 view;
	view = glm::lookAt(glm::vec3(camX, 0.0, camZ), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0)); 

	// TODO: Set up glsl program (at least vertex and fragment shaders). 
	// Hint: Make yourself familiar with the GLSLProgram class (in the file glslprogram.cpp)! 
	// Try to understand what it does and how it is used. 
	// Afterwards, you can ether use it for your purposes or decide to implement your own glsl-program handling.
	prog = new GLSLProgram();
	prog->compileShader("/home/lus0a/CS380/cs380-2021/2_assignment/src/shader/basicVertex.vs");
	prog->compileShader("/home/lus0a/CS380/cs380-2021/2_assignment/src/shader/basicFrag.fs");
	//prog->compileShader("/home/lus0a/CS380/cs380-2021/2_assignment/src/shader/gouraud.vs");
	//prog->compileShader("/home/lus0a/CS380/cs380-2021/2_assignment/src/shader/gouraud.fs");
	//prog->compileShader("/home/lus0a/CS380/cs380-2021/2_assignment/src/shader/phong.vs");
	//prog->compileShader("/home/lus0a/CS380/cs380-2021/2_assignment/src/shader/phong.fs");

	prog->link();
	printf("compile and link success.\n");


	// init objects in the scene
	m_pCube = new VBOCube();
	prog->printActiveAttribs();

	// Once you are done setting up the basic rendering, you can add more complex meshes to your scene.
	// TODO: Add a cylinder and sphere class instead of the VBOCube. Generate their geometry by calculating vertex positions, edges, normals, and texture coordinates. Render them similarly to the cube.
	// 
	// Now cubes and spheres are all nice - but if you want to render anything more complex you will need some kind of CAD model (i.e., essentially a triangle mesh stored in a file). 
	// TODO: Load and render a 'obj' file:

	//m_pMesh = new VBOMesh("/home/lus0a/CS380/cs380-2021/2_assignment/src/data/bs_ears.obj",false,true,true);
}

 
/* TODO: read some background about the framework:

The renderFrame function is called every time we want to render our 3D scene.
Typically we want to render a new frame as a reaction to user input (e.g., mouse dragging the camera), or because we have some animation running in our scene.
We typically aim for 10-120 frames per second (fps) depending on the application (10fps is considered interactive for high demand visualization frameworks, 20fps is usually perceived as fluid, 30fps is for computationally highly demanding gaming, 60fps is the target for gaming, ~120fps is the target for VR).
From these fps-requirements it follows that your renderFrame method is very performance critical. 
It will be called multiple times per second and needs to do all computations necessary to render the scene. 
-> Only compute what you really have to compute in this function (but also not less).

Rendering one frame typically includes:
- updating buffers and variables in reaction to the time that has passed. There are typically three reasons to update something in your scene: 1. animation, 2. physics simulation, 3. user interaction (e.g., camera, render mode, application logic).
- clearing the frame buffer (we typically erase everything we drew the last frame)
- rendering each object in the scene with (a specific) shader program
*/

// render a frame
void renderFrame()
{
	// clear frame buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	// render code goes here
	
	// TODO: use your glsl programs here
	prog->use();
	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);
	glm::mat4 model = glm::mat4(1.0f);
	// TODO: update uniform variables here
	model = glm::translate(model, glm::vec3(va.x, va.y, va.z)); 
	//model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	float objcoef= objsf/10.0;
	model = glm::scale(model, glm::vec3(objcoef, objcoef,objcoef));

	//view = glm::lookAt(vec3(va.x, va.y, va.z), vec3(0.0f, 0.0f, 0.0f), vec3(0,1,0));
	

	projection = glm::perspective(glm::radians(45.0f), (float)gWindowWidth / gWindowHeight, 0.1f, 100.0f);

	float radius = 10.0f;
	float camX = sin(glfwGetTime()) * radius;
	float camZ = cos(glfwGetTime()) * radius;
	view = glm::lookAt(glm::vec3(camX, 0.0, camZ), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));

	prog->setUniform("model", model);
	prog->setUniform("view", view);
	prog->setUniform("projection", projection);
	
	m_pCube->render();
	//m_pMesh->render();
}

void cursor_callback( GLFWwindow* window, double xpos, double ypos )
{   
	cursor_x = xpos;
	cursor_y = ypos;
}       

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	if(yoffset < 0)
	{
		if(objsf > 0) objsf -= 1;
	}
	else
		objsf += 1;
}

void mouse_callback( GLFWwindow* window, int button, int action, int mods )
{
	if ( button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS )
		printf( "Left button is clicked and cursor location is (%f,%f)\n", cursor_x, cursor_y );
	else if ( button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS )
		printf( "Right button is clicked and cursor location is (%f,%f)\n", cursor_x, cursor_y );
}

void key_callback( GLFWwindow* window, int key, int scancode, int action, int mods )
{   
	if ( key == GLFW_KEY_ESCAPE && action == GLFW_PRESS )
	{
		glfwSetWindowShouldClose( window, GLFW_TRUE );
		printf("Key ESC is pressed, the window is closed\n");
	}
	else if ( key == GLFW_KEY_1 && action == GLFW_PRESS )
	{
		shaderOption = 1;
		printf("Switched to Phong lighting + Gouraud shading.\n");
	}
	else if ( key == GLFW_KEY_2 && action == GLFW_PRESS )
	{
		shaderOption = 2;
		printf("Switchde to Phong lighting + Phong shading.\n");
	}
	else if ( key == GLFW_KEY_Q && action == GLFW_PRESS )
		va.x -= 1.0f;
	else if ( key == GLFW_KEY_A && action == GLFW_PRESS )
		va.x += 1.0f;
	else if ( key == GLFW_KEY_W && action == GLFW_PRESS )
		va.y -= 1.0f;
	else if ( key == GLFW_KEY_S && action == GLFW_PRESS )
		va.y += 1.0f;
	else if ( key == GLFW_KEY_E && action == GLFW_PRESS )
		va.z -= 1.0f;
	else if ( key == GLFW_KEY_D && action == GLFW_PRESS )
		va.z += 1.0f;
	else if ( key == GLFW_KEY_R && action == GLFW_PRESS )
	{
		va.x=0;
		va.y=0;
		va.z=1;
	}
}   

// entry point
int main(int argc, char** argv)
{
	
	// set glfw error callback
	glfwSetErrorCallback(glfwErrorCallback);

	// init glfw
	if (!glfwInit()) { 
		exit(EXIT_FAILURE); 
	}

	// init glfw window 
	GLFWwindow* window;
	window = glfwCreateWindow(gWindowWidth, gWindowHeight, "CS380 - OpenGL and GLSL Shaders", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// set GLFW callback functions 
	// TODO: implement and register your callbacks for user interaction
	//glfwSetKeyCallback(window, YOUR_KEY_CALLBACK);
	//glfwSetFramebufferSizeCallback(window, YOUR_FRAMEBUFFER_CALLBACK);
	//glfwSetMouseButtonCallback(window, YOUR_MOUSEBUTTON_CALLBACK);
	//glfwSetCursorPosCallback(window, YOUR_CURSORPOSCALL_BACK);
	//glfwSetScrollCallback(window, YOUR_SCROLL_CALLBACK);
	
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, mouse_callback);
	glfwSetCursorPosCallback(window, cursor_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// make context current (once is sufficient)
	glfwMakeContextCurrent(window);
	
	// get the frame buffer size
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// init the OpenGL API (we need to do this once before any calls to the OpenGL API)
	gladLoadGL();

	// query OpenGL capabilities
	if (!queryGPUCapabilitiesOpenGL()) 
	{
		// quit in case capabilities are insufficient
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// query CUDA capabilities
	if(!queryGPUCapabilitiesCUDA())
	{
		// quit in case capabilities are insufficient
		glfwTerminate();
		exit(EXIT_FAILURE);
	}


	// init our application
	if (!initApplication(argc, argv)) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// setting up our 3D scene
	setupScene();

	// start traversing the main loop
	// loop until the user closes the window 
	while (!glfwWindowShouldClose(window))
	{
		// render one frame  
		renderFrame();

		// swap front and back buffers 
		glfwSwapBuffers(window);

		// poll and process input events (keyboard, mouse, window, ...)
		glfwPollEvents();
	}

	glfwTerminate();
	return EXIT_SUCCESS;
}


