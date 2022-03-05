#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <thread>         
#include <chrono>    
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <glad/glad.h>

#include "Camera.hpp"
#include "Ray.hpp"
#include "kernel.h"
#include "Scene.hpp"
#include "Texture.hpp"
#include "PostProcessing.h"
#include "BVH.hpp"
#include "BMP.hpp"
#include "Definitions.h"
#include "SceneLoader.hpp"


std::thread t;

void startRender(RenderData& data, Scene& scene) {

	RenderParameters pars = data.pars;

	for (int i = 0; i < PASSES_COUNT; i++) {
		if (pars.passes[i]) {
			data.buffers[i] = new float[pars.width * pars.height * 4];
			memset(data.buffers[i], 0, pars.width * pars.height * 4 * sizeof(float));
		}
	}

	data.beautyBuffer = new unsigned char[pars.width * pars.height * 4];

	memset(data.beautyBuffer, 0, pars.width * pars.height * 4 * sizeof(unsigned char));

	renderSetup(&scene);

	t = std::thread(renderCuda, &scene, pars.sampleTarget);

	data.startTime = std::chrono::high_resolution_clock::now();
}

void getRenderData(RenderData& data) {

	int width = data.pars.width;
	int height = data.pars.height;

	int* pathCountBuffer = new int[width * height];

	cudaMemGetInfo(&data.freeMemory, &data.totalMemory);

	getBuffers(data, pathCountBuffer, width * height);

	flipY(data.buffers[BEAUTY], width, height);
	clampPixels(data.buffers[BEAUTY], width, height);
	applysRGB(data.buffers[BEAUTY], width, height);
	HDRtoLDR(data.buffers[BEAUTY], data.beautyBuffer, width, height);

	data.samples = getSamples();
	data.pathCount = 0;

	for (int i = 0; i < width * height; i++)
		data.pathCount += pathCountBuffer[i];

	delete(pathCountBuffer);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}



void checkCompileErrors(GLuint shader, std::string type)
{
	GLint success;
	GLchar infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

int main(int argc, char* argv[])
{
	Scene scene = loadScene(std::string(argv[1]));
	printf("%s\n", argv[1]);
	RenderData data;
	data.pars = RenderParameters(scene.camera.xRes, scene.camera.yRes, atoi(argv[2]));
	startRender(data, scene);

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(1280, 720, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	const char* fragmentShaderSource = "#version 330 core\n"
		"out vec4 FragColor;\n"
		"in vec3 ourColor;\n"
		"in vec2 TexCoord;\n"
		"uniform sampler2D ourTexture;\n"
		"void main()\n"
		"{\n"
		"	FragColor = texture(ourTexture, TexCoord);\n"
		"}\0";

	const char* vertexShaderSource = "#version 330 core\n"
		"layout(location = 0) in vec3 aPos;\n"
		"layout(location = 1) in vec3 aColor;\n"
		"layout(location = 2) in vec2 aTexCoord;\n"
		"out vec3 ourColor;\n"
		"out vec2 TexCoord;\n"
		"void main()\n"
		"{\n"
		"	gl_Position = vec4(aPos, 1.0);\n"
		"	ourColor = aColor;\n"
		"	TexCoord = aTexCoord;\n"
		"}\0";

	unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vertexShaderSource, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");
	// fragment Shader
	unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");

	// shader Program
	unsigned int ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	checkCompileErrors(ID, "PROGRAM");
	// delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertex);
	glDeleteShader(fragment);


	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
	float vertices[] = {
		// positions          // colors           // texture coords
		 1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 0.0f, // top right
		 1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 1.0f, // bottom right
		-1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 1.0f, // bottom left
		-1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 0.0f  // top left 
	};
	unsigned int indices[] = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};
	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);


	// load and create a texture 
	// -------------------------
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// load image, create texture and generate mipmaps

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.pars.width, data.pars.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.beautyBuffer);
	glGenerateMipmap(GL_TEXTURE_2D);


	// render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
			getRenderData(data);

		if (data.samples >= data.pars.sampleTarget - 1) {
			saveBMP(argv[3], data.pars.width, data.pars.height, data.beautyBuffer);
			break;
		}

		auto t2 = std::chrono::high_resolution_clock::now();

		auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - data.startTime);

		printf("\rkPaths/s: %f, %fGB of a total of %fGB used, %d/%d samples. %f seconds running, %d total paths",
			((float)data.pathCount / (float)ms_int.count()),
			(float)(data.totalMemory - data.freeMemory) / (1024 * 1024 * 1024),
			(float)data.totalMemory / (1024 * 1024 * 1024),
			data.samples,
			data.pars.sampleTarget,
			((float)(ms_int).count()) / 1000,
			data.pathCount);

		// input
		// -----
		processInput(window);

		// render
		// ------
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.pars.width, data.pars.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.beautyBuffer);
		glGenerateMipmap(GL_TEXTURE_2D);

		// bind Texture
		glBindTexture(GL_TEXTURE_2D, texture);

		// render container
		glUseProgram(ID);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
		std::this_thread::sleep_for(std::chrono::milliseconds(UPDATE_INTERVAL));
	}

	// optional: de-allocate all resources once they've outlived their purpose:
	// ------------------------------------------------------------------------
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	t.join();
	cudaDeviceReset();
	return 0;
}

/*
int main(int argc, char* argv[]) {

	Scene scene = loadScene(std::string(argv[1]));

	printf("%s\n", argv[1]);

	RenderData data;

	data.pars = RenderParameters(scene.camera.xRes, scene.camera.yRes, atoi(argv[2]));

	sf::RenderWindow window(sf::VideoMode(data.pars.width, data.pars.height, 32), "Render Window");

	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;
	sf::Font font;
	sf::Text text;

	image.create(data.pars.width, data.pars.height);
	texture.loadFromImage(image);
	sprite.setTexture(texture);

	if (!font.loadFromFile("arial.ttf"));
	text.setFont(font);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::Red);

	startRender(data, scene);

	while (window.isOpen()) {

		sf::Event event;

		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}

		getRenderData(data);

		if (data.samples >= data.pars.sampleTarget - 1) {
			saveBMP(argv[3], data.pars.width, data.pars.height, data.beautyBuffer);
			break;
		}

		auto t2 = std::chrono::high_resolution_clock::now();

		auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - data.startTime);

		printf("\rkPaths/s: %f, %fGB of a total of %fGB used, %d/%d samples. %f seconds running, %d total paths",
			((float)data.pathCount / (float)ms_int.count()),
			(float)(data.totalMemory - data.freeMemory) / (1024 * 1024 * 1024),
			(float)data.totalMemory / (1024 * 1024 * 1024),
			data.samples,
			data.pars.sampleTarget,
			((float)(ms_int).count()) / 1000,
			data.pathCount);

		text.setString(std::to_string(data.samples));
		texture.update(data.beautyBuffer);
		window.clear();
		window.draw(sprite);
		window.draw(text);
		window.display();

		std::this_thread::sleep_for(std::chrono::milliseconds(UPDATE_INTERVAL));
	}

	t.join();

	cudaDeviceReset();

	window.close();
	return 0;
}*/