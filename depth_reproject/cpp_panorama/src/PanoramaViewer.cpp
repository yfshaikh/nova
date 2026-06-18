#include "PanoramaViewer.hpp"

#include <cstdio>

bool PanoramaViewer::available_ = false;

bool PanoramaViewer::init(int pano_w, int pano_h, int win_w, int win_h, const char* title) {
    pano_w_ = pano_w;
    pano_h_ = pano_h;

    int argc = 1;
    char arg0[] = "NovaPanoramaGPU";
    char* argv[] = {arg0, nullptr};
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(win_w, win_h);
    glutCreateWindow(title);

    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "glewInit failed\n");
        return false;
    }

    glDisable(GL_DEPTH_TEST);

    // Texture that mirrors the panorama.
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, pano_w_, pano_h_, 0,
                 GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Pixel buffer object the CUDA kernels write into.
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 (size_t)pano_w_ * pano_h_ * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    glutKeyboardFunc(keyCb);
    glutCloseFunc(closeCb);
    glutReshapeFunc(reshapeCb);

    available_ = true;
    return true;
}

bool PanoramaViewer::isAvailable() {
    if (available_)
        glutMainLoopEvent();
    return available_;
}

uchar4* PanoramaViewer::mapBuffer() {
    uchar4* ptr = nullptr;
    size_t bytes = 0;
    if (cudaGraphicsMapResources(1, &cuda_pbo_, 0) != cudaSuccess)
        return nullptr;
    if (cudaGraphicsResourceGetMappedPointer((void**)&ptr, &bytes, cuda_pbo_) != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &cuda_pbo_, 0);
        return nullptr;
    }
    mapped_ = true;
    return ptr;
}

void PanoramaViewer::unmapAndDraw() {
    if (mapped_) {
        cudaGraphicsUnmapResources(1, &cuda_pbo_, 0);
        mapped_ = false;
    }

    glClear(GL_COLOR_BUFFER_BIT);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pano_w_, pano_h_,
                    GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glEnable(GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Flip vertically: panorama row 0 is the top, GL texture origin is bottom.
    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 1.f); glVertex2f(0.f, 0.f);
    glTexCoord2f(1.f, 1.f); glVertex2f(1.f, 0.f);
    glTexCoord2f(1.f, 0.f); glVertex2f(1.f, 1.f);
    glTexCoord2f(0.f, 0.f); glVertex2f(0.f, 1.f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();
    glutPostRedisplay();
}

void PanoramaViewer::setTitle(const char* title) {
    glutSetWindowTitle(title);
}

void PanoramaViewer::close() {
    if (cuda_pbo_) {
        cudaGraphicsUnregisterResource(cuda_pbo_);
        cuda_pbo_ = nullptr;
    }
    if (pbo_) {
        glDeleteBuffers(1, &pbo_);
        pbo_ = 0;
    }
    if (tex_) {
        glDeleteTextures(1, &tex_);
        tex_ = 0;
    }
}

void PanoramaViewer::keyCb(unsigned char key, int, int) {
    if (key == 'q' || key == 'Q' || key == 27)
        available_ = false;
}

void PanoramaViewer::closeCb() {
    available_ = false;
}

void PanoramaViewer::reshapeCb(int w, int h) {
    glViewport(0, 0, w, h);
}
