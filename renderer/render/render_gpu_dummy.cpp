#include "render_gpu.h"

template <typename T>
RendererGpu<T>::RendererGpu(const Camera<T> cam, const Shader<T> shader, Buffer<T> buffer, int wavelength) : BaseRenderer<T>(cam, shader, buffer, wavelength) {
}

template <typename T>
RendererGpu<T>::~RendererGpu() {
}

template <typename T>
void RendererGpu<T>::gpu_to_cpu() {}

template <typename T>
RenderInput<T> RendererGpu<T>::input_to_device(const RenderInput<T> input) { return RenderInput<T>(); }

template <typename T>
void RendererGpu<T>::input_free_device(const RenderInput<T> input) {
  throw std::logic_error("Not implemented");
}

template <typename T>
void RendererGpu<T>::render_mesh(const RenderInput<T> input) {
  throw std::logic_error("Not implemented");
}

template <typename T>
void RendererGpu<T>::render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const unsigned char* pattern, const float* reflectance, const float* camerasensitivity, const float* illumination, int ref_num, int camsen_num, int illu_num, float d_alpha, float d_beta) {
  throw std::logic_error("Not implemented");
}


template class RendererGpu<float>;
