#include <limits>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "render_cpu.h"
#include "common_cpu.h"

template <typename T>
void RendererCpu<T>::render_mesh(RenderInput<T> input) {
  //RenderMeshFunctor<T> functor(input, this->shader, this->cam, this->buffer);
  //iterate_omp_cpu(functor, this->cam.num_pixel(), n_threads);
}

template <typename T>
void RendererCpu<T>::render_mesh_proj(const RenderInput<T> input, const Camera<T> proj, const unsigned char* pattern, const float* reflectance, const float* camerasensitivity, const float* illumination, int ref_num, int camsen_num, int illu_num, float d_alpha, float d_beta) {
  RenderProjectorFunctor<T> functor(input, this->shader, this->cam, proj, pattern, reflectance, camerasensitivity, illumination, d_alpha, d_beta, this->buffer, this->wavelength);
  iterate_omp_cpu(functor, this->cam.num_pixel(), this->n_threads);
}

template class RendererCpu<float>;
