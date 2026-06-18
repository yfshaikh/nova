#pragma once
//
// Minimal CUDA-GL-interop display for a uchar4 BGRA panorama.
// The CUDA kernels write the final image straight into a mapped PBO, which is
// then blitted to a texture and drawn as a fullscreen quad — no CPU copy.
//
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class PanoramaViewer {
public:
    bool init(int pano_w, int pano_h, int win_w, int win_h, const char* title);
    bool isAvailable();        // pumps GLUT events; false once the user quits

    // Map the PBO and return a device pointer the kernels write into (BGRA).
    uchar4* mapBuffer();
    // Unmap, upload PBO -> texture, draw, swap buffers.
    void unmapAndDraw();

    void setTitle(const char* title);
    void close();

    // Returns the last pressed key (besides quit) and clears it; 0 if none.
    int consumeKey();

private:
    int pano_w_ = 0;
    int pano_h_ = 0;
    GLuint pbo_ = 0;
    GLuint tex_ = 0;
    cudaGraphicsResource* cuda_pbo_ = nullptr;
    bool mapped_ = false;

    static bool available_;
    static int last_key_;
    static void keyCb(unsigned char key, int x, int y);
    static void closeCb();
    static void reshapeCb(int w, int h);
};
