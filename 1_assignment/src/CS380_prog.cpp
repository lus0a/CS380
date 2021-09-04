// CS 380 - GPGPU Programming, KAUST
//
// Programming Assignment #1

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

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//cursor position
float cursor_x;
float cursor_y;

// window size
const unsigned int gWindowWidth = 512;
const unsigned int gWindowHeight = 512;


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
	// =============================================================================
	//TODO:
	// for all the following:
	// read up on concepts that you do not know and that are needed here!
	//
	// query and print (to console) OpenGL version and extensions:
	// - query and print GL vendor, renderer, and version using glGetString()
	//
	// query and print GPU OpenGL limits (using glGet(), glGetInteger() etc.):
	// - maximum number of vertex shader attributes
	// - maximum number of varying floats
	// - number of texture image units (in vertex shader and in fragment shader, respectively)
	// - maximum 2D texture size
	// - maximum 3D texture size
	// - maximum number of draw buffers
	// =============================================================================
	const GLubyte *vendor = glGetString( GL_VENDOR );
	const GLubyte *renderer = glGetString( GL_RENDERER );
	const GLubyte *version = glGetString( GL_VERSION );
	
	GLint major, minor;
	glGetIntegerv( GL_MAJOR_VERSION, &major );
	glGetIntegerv( GL_MINOR_VERSION, &minor );

	printf( "----- OpenGL version and extensions ----- \n");
	printf( "GL Vendor   : %s\n", vendor );
	printf( "GL Renderer : %s\n", renderer );
	printf( "GL Version  : %s\n", version );

	GLint nExtensions;
	glGetIntegerv( GL_NUM_EXTENSIONS, &nExtensions);
	for( int i = 0; i < nExtensions; i++ )
		printf( "%s\n", glGetStringi( GL_EXTENSIONS, i ) );
	
	GLint max_vertex_shader_attri, max_varing_floats, num_tex_image_units_vertex, num_tex_image_units_fragment, max_2D_tex, max_3D_tex, max_draw_buf;

	glGetIntegerv( GL_MAX_VERTEX_ATTRIBS, &max_vertex_shader_attri );
	glGetIntegerv( GL_MAX_VARYING_FLOATS, &max_varing_floats );
	glGetIntegerv( GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &num_tex_image_units_vertex );
	glGetIntegerv( GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &num_tex_image_units_fragment );
	glGetIntegerv( GL_MAX_TEXTURE_SIZE, &max_2D_tex );
	glGetIntegerv( GL_MAX_3D_TEXTURE_SIZE, &max_3D_tex );
	glGetIntegerv( GL_MAX_DRAW_BUFFERS, &max_draw_buf );

	printf( "----- GPU OpenGL limits ----- \n");
	printf( "Maximum number of vertex shader attributes	: %d\n", max_vertex_shader_attri );
	printf( "Maximum number of varying floats : %d\n", max_varing_floats );
	printf( "Number of texture image units in fragment shader : %d\n", num_tex_image_units_vertex );
	printf( "Number of texture image units in vertex shader : %d\n", num_tex_image_units_fragment );
	printf( "Maximum 2D texture size : %d\n", max_2D_tex );
	printf( "Maximum 3D texture size : %d\n", max_3D_tex );
	printf( "Maximum number of draw buffers : %d\n", max_draw_buf );

	return true;
}

// query GPU functionality we need for CUDA, return false when not available
bool queryGPUCapabilitiesCUDA()
{
	printf( "----- CUDA functionality ----- \n");

	// Device Count
	int devCount;

	// Get the Device Count
	cudaGetDeviceCount(&devCount);
	
	// Print Device Count
	printf("Device(s): %i\n", devCount);
	
	// =============================================================================
	//TODO:
	// for all the following:
	// read up on concepts that you do not know and that are needed here!
	// 
	// query and print CUDA functionality:
	// - CUDA device properties for every found GPU using cudaGetDeviceProperties():
	//   - device name
	//   - compute capability
	//   - multi-processor count
	//   - clock rate
	//   - total global memory
	//   - shared memory per block
	//   - num registers per block
	//   - warp size (in threads)
	//   - max threads per block
	// =============================================================================

	cudaDeviceProp  prop;
	for (int i = 0; i < devCount; ++i)
	{
		cudaGetDeviceProperties( &prop, 0 );
		printf( "Device Name:  %s\n", prop.name );
		printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
		printf( "Multi-processor count:  %d\n", prop.multiProcessorCount );
		printf( "Clock rate:  %d\n", prop.clockRate );
		printf( "Total global memory:  %lu\n", prop.totalGlobalMem );
		printf( "Shared memory per block:  %lu\n", prop.sharedMemPerBlock );
		printf( "Num registers per block:  %d\n", prop.regsPerBlock );
		printf( "Warp size in threads:  %d\n", prop.warpSize );
		printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock );
	}


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
 

// render a frame
void renderFrame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// render code goes here

}



// =============================================================================
//TODO: read background info about the framework: 
//
//In graphics applications we typically need to create a window where we can display something.
//Window-APIs are quite different on linux, mac, windows and other operating systems. 
//We use GLFW (a cross-platform library) to create a window and to handle the interaction with this window.
//It is a good idea to spend a moment to read up on GLFW:
//https://www.glfw.org/
//
//We will use it to get input events - such as keyboard or mouse events and for displaying frames that have been rendered with OpenGL.
//You should make yourself familiar with the API and read through the code below to understand how it is used to drive a graphics application.
//In general try to understand the anatomy of a typical graphics application!
// =============================================================================

void cursor_callback( GLFWwindow* window, double xpos, double ypos )
{
	cursor_x = xpos;
	cursor_y = ypos;
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
	window = glfwCreateWindow(gWindowWidth, gWindowHeight, "CS380 - GPGPU - OpenGL Window", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// set GLFW callback functions 
	// =============================================================================
	//TODO: read up on certain GLFW callbacks which we will need in the future. 
	//Get an understanding for what a 'callback' is. Questions you should be able to answer include:
	//What is a callback? When is a callback called? How do you use a callback in your application? What are typical examples for callbacks in the context of graphics applications?
	//Have a look at the following examples:
	//
	//glfwSetKeyCallback(window, YOUR_KEY_CALLBACK);
	//glfwSetFramebufferSizeCallback(window, YOUR_FRAMEBUFFER_CALLBACK);
	//glfwSetMouseButtonCallback(window, YOUR_MOUSEBUTTON_CALLBACK);
	//glfwSetCursorPosCallback(window, YOUR_CURSORPOSCALL_BACK);
	//glfwSetScrollCallback(window, YOUR_SCROLL_CALLBACK);
	// ...

	//Implement mouse and keyboard callbacks!
	//Print information about the events on std::cout
	// =============================================================================
	glfwSetCursorPosCallback(window, cursor_callback);	
	glfwSetMouseButtonCallback(window, mouse_callback);
	glfwSetKeyCallback(window, key_callback);

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


