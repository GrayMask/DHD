#ifndef RENDER_CPU_H
#define RENDER_CPU_H

#include "render.h"



template <typename T>
class RendererCpu : public BaseRenderer<T> {
public:
  const int n_threads;

  RendererCpu(const Camera<T> cam, const Shader<T> shader, Buffer<T> buffer, int n_threads, int wavelength) : BaseRenderer<T>(cam, shader, buffer, wavelength), n_threads(n_threads) {
  }

  virtual ~RendererCpu() {
  }

  virtual void render_mesh(const RenderInput<T> input);
  virtual void render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const unsigned char* pattern, const float* reflectance, const float* camerasensitivity, const float* illumination, int ref_num, int camsen_num, int illu_num, float d_alpha, float d_beta);
};

#endif
