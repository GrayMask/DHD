#ifndef RENDER_RENDER_GPU_H
#define RENDER_RENDER_GPU_H

#include "render.h"

template <typename T>
class RendererGpu : public BaseRenderer<T> {
public:
  Buffer<T> buffer_gpu;

  RendererGpu(const Camera<T> cam, const Shader<T> shader, Buffer<T> buffer, int wavelength);

  virtual ~RendererGpu();

  virtual void gpu_to_cpu();
  virtual RenderInput<T> input_to_device(const RenderInput<T> input);
  virtual void input_free_device(const RenderInput<T> input);

  virtual void render_mesh(const RenderInput<T> input);
  virtual void render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const unsigned char* pattern, const float* reflectance, const float* camerasensitivity, const float* illumination, int ref_num, int camsen_num, int illu_num, float d_alpha, float d_beta);
};


#endif
