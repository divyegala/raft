/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <raft/error.hpp>

#include <cusparse_v2.h>
///@todo: enable this once logging is enabled
//#include <cuml/common/logger.hpp>

#define _CUSPARSE_ERR_TO_STR(err) \
  case err:                       \
    return #err;

namespace raft {

/**
 * @brief Exception thrown when a cuSparse error is encountered.
 */
struct cusparse_error : public raft::exception {
  explicit cusparse_error(char const* const message)
    : raft::exception(message) {}
  explicit cusparse_error(std::string const& message)
    : raft::exception(message) {}
};

namespace sparse {
namespace detail {

inline const char* cusparse_error_to_string(cusparseStatus_t err) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10100
  return cusparseGetErrorString(status);
#else   // CUDART_VERSION
  switch (err) {
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_SUCCESS);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_NOT_INITIALIZED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ALLOC_FAILED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INVALID_VALUE);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_ARCH_MISMATCH);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_EXECUTION_FAILED);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_INTERNAL_ERROR);
    _CUSPARSE_ERR_TO_STR(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    default:
      return "CUSPARSE_STATUS_UNKNOWN";
  };
#endif  // CUDART_VERSION
}

}  // namespace detail
}  // namespace sparse
}  // namespace raft

#undef _CUSPARSE_ERR_TO_STR

/**
 * @brief Error checking macro for cuSparse runtime API functions.
 *
 * Invokes a cuSparse runtime API function call, if the call does not return
 * CUSPARSE_STATUS_SUCCESS, throws an exception detailing the cuSparse error that occurred
 */
#define CUSPARSE_TRY(call)                                                   \
  do {                                                                       \
    cusparseStatus_t const status = (call);                                  \
    if (CUSPARSE_STATUS_SUCCESS != status) {                                 \
      std::string msg{};                                                     \
      SET_ERROR_MSG(msg, "cuSparse error encountered at: ",                  \
                    "call='%s', Reason=%d:%s", #call, status,                \
                    raft::sparse::detail::cusparse_error_to_string(status)); \
      throw raft::cusparse_error(msg);                                       \
    }                                                                        \
  } while (0)

/** FIXME: temporary alias for cuML compatibility */
#define CUSPARSE_CHECK(call) CUSPARSE_TRY(call)

//@todo: enable this once logging is enabled
#if 0
/** check for cusparse runtime API errors but do not assert */
#define CUSPARSE_CHECK_NO_THROW(call)                                          \
  do {                                                                         \
    cusparseStatus_t err = call;                                               \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                      \
      CUML_LOG_ERROR("CUSPARSE call='%s' got errorcode=%d err=%s", #call, err, \
                     raft::sparse::detail::cusparse_error_to_string(err));     \
    }                                                                          \
  } while (0)
#endif

namespace raft {
namespace sparse {

/**
 * @defgroup gthr cusparse gather methods
 * @{
 */
template <typename T>
cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz, const T* vals,
                              T* vals_sorted, int* d_P, cudaStream_t stream);
template <>
inline cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz,
                                     const double* vals, double* vals_sorted,
                                     int* d_P, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseDgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}
template <>
inline cusparseStatus_t cusparsegthr(cusparseHandle_t handle, int nnz,
                                     const float* vals, float* vals_sorted,
                                     int* d_P, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  return cusparseSgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}
/** @} */

/**
 * @defgroup coo2csr cusparse COO to CSR converter methods
 * @{
 */
template <typename T>
void cusparsecoo2csr(cusparseHandle_t handle, const T* cooRowInd, int nnz,
                     int m, T* csrRowPtr, cudaStream_t stream);
template <>
inline void cusparsecoo2csr(cusparseHandle_t handle, const int* cooRowInd,
                            int nnz, int m, int* csrRowPtr,
                            cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr,
                                  CUSPARSE_INDEX_BASE_ZERO));
}
/** @} */

/**
 * @defgroup coosort cusparse coo sort methods
 * @{
 */
template <typename T>
size_t cusparsecoosort_bufferSizeExt(  // NOLINT
  cusparseHandle_t handle, int m, int n, int nnz, const T* cooRows,
  const T* cooCols, cudaStream_t stream);
template <>
inline size_t cusparsecoosort_bufferSizeExt(  // NOLINT
  cusparseHandle_t handle, int m, int n, int nnz, const int* cooRows,
  const int* cooCols, cudaStream_t stream) {
  size_t val;
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(
    cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRows, cooCols, &val));
  return val;
}

template <typename T>
void cusparsecoosortByRow(  // NOLINT
  cusparseHandle_t handle, int m, int n, int nnz, T* cooRows, T* cooCols, T* P,
  void* pBuffer, cudaStream_t stream);
template <>
inline void cusparsecoosortByRow(  // NOLINT
  cusparseHandle_t handle, int m, int n, int nnz, int* cooRows, int* cooCols,
  int* P, void* pBuffer, cudaStream_t stream) {
  CUSPARSE_CHECK(cusparseSetStream(handle, stream));
  CUSPARSE_CHECK(
    cusparseXcoosortByRow(handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}
/** @} */

/**
 * @defgroup Gemmi cusparse gemmi operations
 * @{
 */
inline cusparseStatus_t cusparsegemmi(
  cusparseHandle_t handle, int m, int n, int k, int nnz, const float* alpha,
  const float* A, int lda, const float* cscValB, const int* cscColPtrB,
  const int* cscRowIndB, const float* beta, float* C, int ldc) {
  return cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}
inline cusparseStatus_t cusparsegemmi(
  cusparseHandle_t handle, int m, int n, int k, int nnz, const double* alpha,
  const double* A, int lda, const double* cscValB, const int* cscColPtrB,
  const int* cscRowIndB, const double* beta, double* C, int ldc) {
  return cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}
/** @} */

}  // namespace sparse
}  // namespace raft
